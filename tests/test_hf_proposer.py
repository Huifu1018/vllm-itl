import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

from vllm_itl.hf_proposer import _clone_nested_tensors


class HFProposerTests(unittest.TestCase):
    @unittest.skipIf(torch is None, "torch is not installed")
    def test_nested_tensor_clone_does_not_alias(self):
        source = ((torch.tensor([1, 2]), torch.tensor([3])),)
        cloned = _clone_nested_tensors(source)
        source[0][0][0] = 99

        self.assertEqual(int(cloned[0][0][0]), 1)


if __name__ == "__main__":
    unittest.main()
