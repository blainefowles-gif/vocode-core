import json
import unittest
from unittest.mock import patch

import app
from riteway_knowledge import CATALOG_PATH, build_agent_instructions, load_catalog


class RitewayKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()

    def test_catalog_contains_all_orderable_website_products(self):
        self.assertEqual(self.catalog["source_product_count"], 57)
        self.assertEqual(self.catalog["orderable_product_count"], 54)
        self.assertEqual(len(self.catalog["products"]), 54)
        self.assertEqual(
            len({product["slug"] for product in self.catalog["products"]}), 54
        )

    def test_catalog_uses_current_public_prices(self):
        products = {product["slug"]: product for product in self.catalog["products"]}
        self.assertEqual(products["pea-gravel"]["delivery_price"], 35)
        self.assertEqual(products["pea-gravel"]["pickup_price"], 39)
        self.assertEqual(products["three-eighth-minus-fines"]["delivery_price"], 14)
        self.assertEqual(products["three-eighth-minus-fines"]["pickup_price"], 18)

    def test_prompt_contains_business_delivery_and_inventory_rules(self):
        prompt = build_agent_instructions(
            self.catalog,
            inventory={"pea-gravel": {"inStock": True}},
            caller_phone="+14355550123",
        )
        self.assertIn("virtual phone receptionist", prompt)
        self.assertIn("Five yards or less to a main service area", prompt)
        self.assertIn("Rock capacity is 16 yards", prompt)
        self.assertIn("Soil, dirt, mulch, and compost capacity is 17 yards", prompt)
        self.assertIn("Pea Gravel", prompt)
        self.assertIn("currently marks this in stock", prompt)
        self.assertIn("Do not use the old $7-per-mile rule", prompt)
        self.assertIn("+14355550123", prompt)


class RitewayBridgeConfigurationTests(unittest.TestCase):
    def test_session_update_uses_current_realtime_schema_and_pcmu(self):
        payload = app.build_session_update("test instructions")
        session = payload["session"]
        self.assertEqual(payload["type"], "session.update")
        self.assertEqual(session["type"], "realtime")
        self.assertEqual(session["output_modalities"], ["audio"])
        self.assertEqual(session["audio"]["input"]["format"]["type"], "audio/pcmu")
        self.assertEqual(session["audio"]["output"]["format"]["type"], "audio/pcmu")
        self.assertTrue(
            session["audio"]["input"]["turn_detection"]["interrupt_response"]
        )
        self.assertEqual(session["tools"][0]["name"], "record_inquiry")
        self.assertNotIn("modalities", session)
        self.assertNotIn("input_audio_format", session)

    def test_twiml_contains_stream_auth_and_escaped_caller(self):
        with patch.object(app, "MEDIA_STREAM_TOKEN", "secret&token"):
            twiml = app.build_twiml('+1<435>"555"')
        self.assertIn("wss://riteway-ai-agent.onrender.com/media", twiml)
        self.assertIn('name="token" value="secret&amp;token"', twiml)
        self.assertIn("+1&lt;435&gt;&quot;555&quot;", twiml)

    def test_catalog_json_is_valid_json(self):
        with CATALOG_PATH.open(encoding="utf-8") as file:
            parsed = json.load(file)
        self.assertEqual(parsed["orderable_product_count"], 54)


if __name__ == "__main__":
    unittest.main()
