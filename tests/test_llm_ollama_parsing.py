import sys
import types
import unittest
from unittest.mock import MagicMock, patch

fake_tenacity = types.ModuleType("tenacity")
fake_tenacity.retry = lambda *a, **k: (lambda f: f)
fake_tenacity.stop_after_attempt = lambda *a, **k: None
fake_tenacity.wait_exponential = lambda *a, **k: None
fake_tenacity.retry_if_exception_type = lambda *a, **k: None
fake_tenacity.before_sleep_log = lambda *a, **k: None
fake_tenacity.retry_if_exception = lambda *a, **k: None
sys.modules.setdefault("tenacity", fake_tenacity)

fake_weaviate = types.ModuleType("weaviate")
fake_util = types.ModuleType("weaviate.util")
fake_util.generate_uuid5 = lambda identifier, namespace=None: "00000000-0000-0000-0000-000000000000"
fake_weaviate.util = fake_util
sys.modules.setdefault("weaviate", fake_weaviate)
sys.modules.setdefault("weaviate.util", fake_util)

fake_filelock = types.ModuleType("filelock")
class _DummyLock:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False
fake_filelock.FileLock = _DummyLock
fake_filelock.Timeout = Exception
sys.modules.setdefault("filelock", fake_filelock)

fake_json_repair = types.ModuleType("json_repair")
import re
fake_json_repair.repair_json = lambda s: re.sub(r",\s*([}\]])", r"\1", s)
sys.modules["json_repair"] = fake_json_repair

fake_torch = types.ModuleType("torch")
class _Cuda:
    @staticmethod
    def is_available():
        return False
class _Mps:
    @staticmethod
    def is_available():
        return False
fake_torch.cuda = _Cuda()
fake_torch.backends = types.SimpleNamespace(mps=_Mps())
sys.modules.setdefault("torch", fake_torch)

fake_metrics = types.ModuleType("metrics")
class _Metric:
    def labels(self, **kwargs):
        return self
    def inc(self):
        return None
    def observe(self, _val):
        return None
fake_metrics.LLM_CALLS = _Metric()
fake_metrics.LLM_LATENCY = _Metric()
sys.modules.setdefault("metrics", fake_metrics)

import llm_interaction


class TestOllamaParsing(unittest.TestCase):
    def setUp(self):
        self.manager = llm_interaction.LLMManager(api_keys=[], model_name="dummy")
        self.manager.llm_mode = "local"

    @patch("llm_interaction.requests.post")
    def test_empty_response_retries_and_errors(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"response": "\n"}
        mock_post.return_value = resp

        result = self.manager._generate_content_with_retry("prompt", max_output_tokens=16)
        self.assertTrue(result.startswith("Error:"))
        self.assertGreaterEqual(mock_post.call_count, 2)

    @patch("llm_interaction.requests.post")
    def test_message_content_fallback(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"message": {"content": "OK"}}
        mock_post.return_value = resp

        result = self.manager._call_ollama("prompt")
        self.assertEqual(result, "OK")

    @patch("llm_interaction.requests.post")
    def test_error_field_is_failure(self, mock_post):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"error": "model overloaded"}
        mock_post.return_value = resp

        result = self.manager._call_ollama("prompt")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
