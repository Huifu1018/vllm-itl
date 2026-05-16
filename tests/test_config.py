import unittest
from unittest.mock import patch

from vllm_itl.config import TokenITLConfig, TokenTimingConfig


class ConfigTests(unittest.TestCase):
    def test_tokentiming_config_validation(self):
        with self.assertRaises(ValueError):
            TokenTimingConfig(num_draft_tokens=0).validate()

    def test_env_config_can_disable_sampling(self):
        with patch.dict("os.environ", {"VLLM_ITL_ALLOW_SAMPLING": "0"}):
            config = TokenITLConfig.from_env()

        self.assertFalse(config.allow_sampling)

    def test_env_config_can_disable_metrics(self):
        with patch.dict("os.environ", {"VLLM_ITL_METRICS_LOG_INTERVAL": "0"}):
            config = TokenITLConfig.from_env()

        self.assertIsNone(config.metrics_log_interval)

    def test_draft_tp_rank_from_env(self):
        with patch.dict("os.environ", {"VLLM_ITL_DRAFT_TP_RANK": "2"}):
            config = TokenITLConfig.from_env()

        self.assertEqual(config.draft_tp_rank, 2)

    def test_draft_tp_rank_rejects_negative_values(self):
        with patch.dict("os.environ", {"VLLM_ITL_DRAFT_TP_RANK": "-1"}):
            with self.assertRaises(ValueError):
                TokenITLConfig.from_env()


if __name__ == "__main__":
    unittest.main()
