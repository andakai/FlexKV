import time
from typing import NewType, Optional

import numpy as np

from flexkv import c_ext


HashType = NewType('HashType', int)

def get_hash_size() -> int:
    return int(c_ext.get_hash_size())

class Hasher:
    def __init__(self) -> None:
        self.hasher = c_ext.Hasher()

    def reset(self) -> None:
        self.hasher.reset()

    def update(self, array: np.ndarray) -> None:
        self.hasher.update_numpy(np.ascontiguousarray(array))

    def digest(self) -> HashType:
        return HashType(self.hasher.digest())

_HASHER = Hasher()

def hash_array(array: np.ndarray) -> HashType:
    _HASHER.reset()
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def hash_array_with_prefix(array: np.ndarray, prefix: int) -> HashType:
    """Hash array with a prefix value (e.g., device type) to avoid collisions."""
    _HASHER.reset()
    # Add prefix as a single-element array to the hash
    _HASHER.update(np.array([prefix], dtype=np.int64))
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def gen_hashes(token_ids: np.ndarray, tokens_per_block: int, hasher: Optional[Hasher] = None,
               num_hash_lookahead_tokens: int = 0) -> np.ndarray:
    """Hash complete KV blocks together with their subsequent token context."""
    token_ids = np.ascontiguousarray(token_ids)
    if token_ids.dtype != np.int64:
        raise TypeError(f"gen_hashes: token_ids must be int64, got {token_ids.dtype}")
    if num_hash_lookahead_tokens < 0:
        raise ValueError("num_hash_lookahead_tokens must be nonnegative")
    if tokens_per_block <= 0:
        raise ValueError("tokens_per_block must be positive")
    if hasher is None:
        hasher = Hasher()
    if num_hash_lookahead_tokens:
        # Version and geometry separate these keys from the legacy token chain.
        hasher.update(np.array([-1, 1, tokens_per_block, num_hash_lookahead_tokens], dtype=np.int64))
        hasher.update(token_ids[:num_hash_lookahead_tokens])
        token_ids = token_ids[num_hash_lookahead_tokens:]
    block_hashes = np.zeros(token_ids.size // tokens_per_block, dtype=np.uint64)
    c_ext.gen_hashes_numpy(hasher.hasher, token_ids, tokens_per_block, block_hashes)
    return block_hashes

if __name__ == "__main__":
    np.random.seed(0)
    token_ids = np.random.randint(0, 10000, (1000, ), dtype=np.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    result = hash_array(token_ids)
    start = time.time()
    for i in range(1):
        result = hash_array(token_ids)
    end = time.time()
    print(f"array hash: {result}, average time: {(end - start)*1000/5}ms")
    # start = time.time()
    # result2 = gen_hashes(token_ids, 16)
    # end = time.time()
    # print(f"block hashes: {result2}, time: {(end - start)*1000}ms")
