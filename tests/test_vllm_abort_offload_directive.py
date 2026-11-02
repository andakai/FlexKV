from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import numpy as np

from vllm.v1.engine import FinishReason

from flexkv.integration.vllm.vllm_v1_adapter import FlexKVSchedulerConnector
from flexkv.common.block import SequenceMeta


def _connector() -> FlexKVSchedulerConnector:
    connector = object.__new__(FlexKVSchedulerConnector)
    connector.req_id_to_task_dict = {}
    connector.maybe_skip_put = False
    connector.block_size = 16
    connector.num_hash_lookahead_tokens = 0
    connector.flexkv_stats = Mock()
    connector._put_match = Mock(return_value=(-1, 0, 0))
    return connector


def _request(reason: FinishReason, offload: bool):
    return SimpleNamespace(
        request_id="request-0",
        is_finished=lambda: True,
        get_finished_reason=lambda: reason,
        offload_kv_on_finish=offload,
        num_tokens=32,
    )


def test_abort_without_vllm_directive_does_not_enter_put_path():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.ABORT, offload=False), [0, 1]
    )

    assert delayed is False
    connector._put_match.assert_not_called()


def test_abort_with_vllm_directive_enters_existing_put_path():
    connector = _connector()

    request = _request(FinishReason.ABORT, offload=True)
    delayed = connector.request_finished(request, [0, 1])

    assert delayed is False
    connector._put_match.assert_called_once_with(
        request=request,
        num_tokens_to_put=16,
    )
    connector.flexkv_stats.record_put.assert_called_once_with(
        num_all_tokens=16,
        num_unmatched_tokens=0,
    )


def test_normal_finish_remains_eligible_for_put():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.STOP, offload=False), [0, 1]
    )

    assert delayed is False
    connector._put_match.assert_called_once()
    connector.flexkv_stats.record_put.assert_called_once()


@pytest.mark.parametrize(
    "lookahead,computed,in_flight,stale,expected",
    [(1, 32, 0, 0, 32), (3, 32, 0, 0, 16),
     (1, 32, 1, 0, 16), (1, 32, 17, 16, 16)],
)
def test_lookahead_finish_saves_only_complete_kv_with_hash_context(
    lookahead, computed, in_flight, stale, expected
):
    connector = _connector()
    connector.num_hash_lookahead_tokens = lookahead
    request = _request(FinishReason.ABORT, offload=True)
    request.num_tokens = 33
    request.num_computed_tokens = computed
    request.num_in_flight_tokens = in_flight
    request.num_stale_output_tokens = stale
    connector.request_finished(request, [0, 1, 2])
    connector._put_match.assert_called_once_with(
        request=request, num_tokens_to_put=expected,
    )


def test_abort_with_no_computed_blocks_skips_put():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.ABORT, offload=True), []
    )

    assert delayed is False
    connector._put_match.assert_not_called()
    connector.flexkv_stats.record_put.assert_not_called()


def test_abort_limits_put_to_computed_blocks():
    connector = _connector()
    request = _request(FinishReason.ABORT, offload=True)
    request.num_tokens = 80

    connector.request_finished(request, [4, 5])

    connector._put_match.assert_called_once_with(
        request=request,
        num_tokens_to_put=32,
    )
    connector.flexkv_stats.record_put.assert_called_once_with(
        num_all_tokens=32,
        num_unmatched_tokens=0,
    )


def test_lookahead_lookup_and_store_isolate_lora_and_mm_positions():
    connector = _connector()
    del connector._put_match  # Exercise the real adapter's store key path.
    connector.num_hash_lookahead_tokens = 1
    connector.cache_namespace = "model-and-draft"
    connector.flexkv_manager = Mock()
    connector.flexkv_manager.get_match.return_value = (-1, np.zeros(33, dtype=bool))
    connector.flexkv_manager.put_match.return_value = (-1, np.zeros(33, dtype=bool))
    keys = []
    for path, local_id, offset, length, salt, user_namespace in [
        ("adapter-a", 1, 0, 4, None, None),
        ("adapter-b", 1, 0, 4, None, None),
        ("adapter-a", 2, 0, 4, None, None),
        ("adapter-a", 1, 4, 4, None, None),
        ("adapter-a", 1, 0, 8, None, None),
        (None, 0, 0, 4, '["adapter", "adapter-a"]', None),
        (None, 0, 0, 4, '["lora", "adapter", "adapter-a"]', None),
        (None, 0, 0, 4, None, ['["lora", "adapter", "adapter-a"]']),
    ]:
        request = _request(FinishReason.STOP, offload=False)
        request.num_tokens = 33
        request.num_computed_tokens = 32
        request.num_in_flight_tokens = request.num_stale_output_tokens = 0
        request.all_token_ids = list(range(33))
        request.prompt_embeds = None
        request.lora_request = (SimpleNamespace(
            lora_name="adapter", lora_path=path, lora_int_id=local_id)
            if path is not None else None)
        request.cache_salt = salt
        request.namespace_info = user_namespace
        request.mm_features = [SimpleNamespace(
            identifier="image-a",
            mm_position=SimpleNamespace(offset=offset, length=length))]
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        assert connector.request_finished(request, [0, 1]) is False
        lookup = connector.flexkv_manager.get_match.call_args.kwargs
        store = connector.flexkv_manager.put_match.call_args.kwargs
        assert lookup["namespace"] == store["namespace"]
        np.testing.assert_array_equal(lookup["token_ids"], store["token_ids"])
        np.testing.assert_array_equal(lookup["token_mask"], store["token_mask"])
        keys.append(tuple(SequenceMeta(
            token_ids=store["token_ids"], tokens_per_block=16,
            namespace=store["namespace"], num_hash_lookahead_tokens=1,
        ).block_hashes))
    assert keys[0] == keys[2]
    assert len(set(keys)) == 7
