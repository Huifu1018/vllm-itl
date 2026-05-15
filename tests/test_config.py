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


if __name__ == "__main__":
    unittest.main()
