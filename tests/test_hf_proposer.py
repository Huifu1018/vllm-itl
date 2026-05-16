import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vllm_itl.hf_proposer import _clone_cache_for_reuse, _clone_nested_tensors


class HFProposerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_nested_tensor_clone_does_not_alias(self):
        source = ((torch.tensor([1, 2]), torch.tensor([3])),)
        cloned = _clone_nested_tensors(source)
        source[0][0][0] = 99

        self.assertEqual(int(cloned[0][0][0]), 1)

    def test_cache_object_clone_keeps_cache_api(self):
        class FakeCache:
            def __init__(self):
                self.values = [1]

            def get_seq_length(self):
                return len(self.values)

            def to_legacy_cache(self):
                return ("legacy",)

        cache = FakeCache()
        cloned = _clone_cache_for_reuse(cache)
        cache.values.append(2)

        self.assertNotIsInstance(cloned, tuple)
        self.assertEqual(cloned.get_seq_length(), 1)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_cache_object_fallback_rebuilds_from_legacy_cache(self):
        class FakeCache:
            def __init__(self, legacy_cache=None):
                self.legacy_cache = legacy_cache

            def __deepcopy__(self, memo):
                raise TypeError("force legacy fallback")

            def get_seq_length(self):
                return 1

            def to_legacy_cache(self):
                return ((torch.tensor([1]),),)

            @classmethod
            def from_legacy_cache(cls, legacy_cache):
                return cls(legacy_cache=legacy_cache)

        cloned = _clone_cache_for_reuse(FakeCache())

        self.assertIsInstance(cloned, FakeCache)
        self.assertEqual(int(cloned.legacy_cache[0][0][0]), 1)


if __name__ == "__main__":
    unittest.main()
