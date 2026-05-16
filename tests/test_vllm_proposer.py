import types
import unittest

from vllm_itl.vllm.proposer import VllmTokenITLProposer, _TPRuntime


class VllmProposerTests(unittest.TestCase):
    def test_export_payload_normalizes_token_ids(self):
        proposer = VllmTokenITLProposer.__new__(VllmTokenITLProposer)

        payload = proposer._export_payload([[1, "2"], []])

        self.assertEqual(payload, {"draft_token_ids": [[1, 2], []]})

    def test_apply_payload_normalizes_token_ids(self):
        proposer = VllmTokenITLProposer.__new__(VllmTokenITLProposer)

        draft_ids = proposer._apply_payload({"draft_token_ids": [["3"], [4]]})

        self.assertEqual(draft_ids, [[3], [4]])

    def test_broadcast_uses_configured_tp_source_rank(self):
        class FakeGroup:
            def broadcast_object(self, payload, src=0):
                return {"payload": payload, "src": src}

        proposer = VllmTokenITLProposer.__new__(VllmTokenITLProposer)
        proposer.tp_runtime = _TPRuntime(rank=1, world_size=2, group=FakeGroup())
        proposer.config = types.SimpleNamespace(draft_tp_rank=1)

        result = proposer._broadcast_payload({"draft_token_ids": [[1]]})

        self.assertEqual(result["src"], 1)
        self.assertEqual(result["payload"], {"draft_token_ids": [[1]]})


if __name__ == "__main__":
    unittest.main()
