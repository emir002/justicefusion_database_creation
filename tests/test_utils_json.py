import sys
import types
import unittest

fake_weaviate = types.ModuleType("weaviate")
fake_util = types.ModuleType("weaviate.util")
fake_util.generate_uuid5 = lambda identifier, namespace=None: "00000000-0000-0000-0000-000000000000"
fake_weaviate.util = fake_util
sys.modules.setdefault("weaviate", fake_weaviate)
sys.modules.setdefault("weaviate.util", fake_util)

fake_json_repair = types.ModuleType("json_repair")
import re
fake_json_repair.repair_json = lambda s: re.sub(r",\s*([}\]])", r"\1", s)
sys.modules["json_repair"] = fake_json_repair

import utils


class TestSafeJsonLoads(unittest.TestCase):
    def test_plain_valid_json(self):
        parsed = utils.safe_json_loads('{"nodes": [], "edges": []}', source_info="plain")
        self.assertEqual(parsed, {"nodes": [], "edges": []})

    def test_fenced_json(self):
        payload = '```json\n{"nodes": [{"entity_name": "X"}], "edges": []}\n```'
        parsed = utils.safe_json_loads(payload, source_info="fenced")
        self.assertEqual(parsed["nodes"][0]["entity_name"], "X")

    def test_json_with_surrounding_text(self):
        payload = 'Here is data:\n{"nodes": [], "edges": []}\nthanks.'
        parsed = utils.safe_json_loads(payload, source_info="surrounded")
        self.assertEqual(parsed, {"nodes": [], "edges": []})

    def test_malformed_json_repair(self):
        payload = '{"nodes": [], "edges": [],}'
        parsed = utils.safe_json_loads(payload, source_info="repair")
        self.assertIsInstance(parsed, dict)


if __name__ == "__main__":
    unittest.main()
