import unittest
from unittest.mock import patch
from pathlib import Path

from backend.app.services.vision_service import verify_evidence


class TestVisionService(unittest.TestCase):

    def test_missing_image_path_returns_neutral_status(self):
        # Critical rule: Missing vision evidence is NOT negative evidence
        result = verify_evidence(image_path=None)
        self.assertFalse(result["available"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "NO_IMAGE_EVIDENCE")
        self.assertIsNone(result["damage_detected"])
        self.assertIsNone(result["confidence"])

    def test_empty_image_string_returns_neutral_status(self):
        result = verify_evidence(image_path="")
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "NO_IMAGE_EVIDENCE")

    def test_nonexistent_image_path_handled_safely(self):
        result = verify_evidence(image_path="nonexistent/path/to/evidence.jpg")
        self.assertFalse(result["available"])
        self.assertFalse(result["verified"])
        self.assertEqual(result["reason"], "IMAGE_NOT_FOUND")

    def test_missing_credentials_handled_gracefully(self):
        dummy_path = "data/demo/.gitkeep"  # an existing file
        with patch("vision.image_analyzer.analyze_image", side_effect=RuntimeError("GEMINI_API_KEY is not configured")):
            result = verify_evidence(image_path=dummy_path)
            self.assertFalse(result["available"])
            self.assertFalse(result["verified"])
            self.assertEqual(result["reason"], "VISION_CREDENTIALS_UNAVAILABLE")

    def test_successful_mocked_vision_analysis(self):
        dummy_path = "data/demo/.gitkeep"
        mock_output = {
            "image_quality": "GOOD",
            "product_condition": "DAMAGED",
            "damage_detected": True,
            "packaging_condition": "DAMAGED",
            "evidence_consistent": True,
            "confidence": 0.94,
            "explanation": "Headphones show cracked headband and physical damage.",
            "image_path": dummy_path,
        }

        with patch("vision.image_analyzer.analyze_image", return_value=mock_output):
            result = verify_evidence(image_path=dummy_path, return_reason="Item arrived broken")
            self.assertTrue(result["available"])
            self.assertTrue(result["verified"])
            self.assertTrue(result["damage_detected"])
            self.assertTrue(result["evidence_consistent"])
            self.assertEqual(result["confidence"], 0.94)
            self.assertEqual(result["product_condition"], "DAMAGED")


if __name__ == "__main__":
    unittest.main()
