import sys
import types
import unittest
from unittest.mock import patch

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


class TestClassificationAndTitleFixups(unittest.TestCase):
    def setUp(self):
        self.manager = llm_interaction.LLMManager(api_keys=[], model_name="dummy")
        self.manager.llm_mode = "local"

    def test_pure_law_heuristic_override_to_law(self):
        snippet = "Član 58\n(1) Naručilac pokreće postupak javne nabavke.\n(2) Postupak se vodi u skladu sa ovim zakonom."
        label = self.manager.classify_document(snippet, filename="Zakon o javnim nabavkama član 58 stav 2.txt")
        self.assertEqual(label, "LAW")

    def test_law_plus_commentary_is_mixed(self):
        snippet = "Član 58\n(1) Naručilac pokreće postupak.\nKomentar: Ovo pojašnjenje tumačenje člana je dato radi prakse."
        with patch.object(self.manager, "_generate_content_with_retry", return_value="MIXED") as mocked:
            label = self.manager.classify_document(snippet, filename="Zakon o javnim nabavkama član 58 stav 2.txt")
        self.assertEqual(label, "MIXED")
        mocked.assert_called()

    def test_other_snippet_is_other(self):
        snippet = "Dnevni sportski pregled i vremenska prognoza za narednu sedmicu."
        with patch.object(self.manager, "_generate_content_with_retry", return_value="OTHER"):
            label = self.manager.classify_document(snippet, filename="vijesti.txt")
        self.assertEqual(label, "OTHER")

    def test_title_typo_fix_nabavkam(self):
        snippet = "Zakon o javnim nabavkama\nČlan 58"
        with patch.object(self.manager, "_generate_content_with_retry", return_value="Zakon o javnim nabavkam"):
            title = self.manager.extract_document_title(snippet, original_filename="doc.txt")
        self.assertEqual(title, "Zakon o javnim nabavkama")


if __name__ == "__main__":
    unittest.main()
