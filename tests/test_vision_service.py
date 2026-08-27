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

    def test_vision_analyze_api_upload_endpoint(self):
        from fastapi.testclient import TestClient
        from backend.app.main import app

        client = TestClient(app)

        # 1. Test unsupported file format
        res_bad = client.post(
            "/api/v1/vision/analyze",
            files={"file": ("test.pdf", b"NOT_AN_IMAGE", "application/pdf")},
        )
        self.assertEqual(res_bad.status_code, 400)
        self.assertIn("Unsupported image format", res_bad.json()["detail"])

        # 2. Test valid image upload
        mock_output = {
            "image_quality": "EXCELLENT",
            "product_condition": "CRACKED_SCREEN",
            "damage_detected": True,
            "packaging_condition": "INTACT",
            "evidence_consistent": True,
            "confidence": 0.96,
            "explanation": "Visual inspection shows screen fracture.",
            "image_path": "uploaded_test.png",
        }
        with patch("vision.image_analyzer.analyze_image", return_value=mock_output):
            res_ok = client.post(
                "/api/v1/vision/analyze",
                files={"file": ("test_damage.png", b"\x89PNG\r\n\x1a\nFakeImageData", "image/png")},
                data={"return_reason": "Broken screen"},
            )
            self.assertEqual(res_ok.status_code, 200)
            data = res_ok.json()
            self.assertTrue(data["available"])
            self.assertTrue(data["verified"])
            self.assertTrue(data["damage_detected"])
            self.assertEqual(data["confidence"], 0.96)


if __name__ == "__main__":
    unittest.main()
