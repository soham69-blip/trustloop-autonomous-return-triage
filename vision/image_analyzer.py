from pathlib import Path
from typing import Optional
import json
import os

from PIL import Image
from dotenv import load_dotenv
from google import genai


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

SUPPORTED_FORMATS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# GEMINI CLIENT
# ============================================================

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not configured.\n"
        "Add it to the project's .env file."
    )

client = genai.Client(
    api_key=API_KEY
)


# ============================================================
# IMAGE VALIDATION
# ============================================================

def validate_image(image_path: str) -> Path:

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported image format: {path.suffix}\n"
            f"Supported formats: "
            f"{sorted(SUPPORTED_FORMATS)}"
        )

    try:
        with Image.open(path) as image:
            image.verify()

    except Exception as exc:
        raise ValueError(
            f"Invalid or corrupted image: {path}"
        ) from exc

    return path


# ============================================================
# IMAGE METADATA
# ============================================================

def get_image_metadata(
    image_path: str
) -> dict:

    path = validate_image(image_path)

    with Image.open(path) as image:

        return {
            "filename": path.name,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "mode": image.mode,
            "file_size_bytes": path.stat().st_size,
        }


# ============================================================
# GEMINI VISION ANALYSIS
# ============================================================

def analyze_image(
    image_path: str,
    return_reason: Optional[str] = None,
) -> dict:

    path = validate_image(image_path)

    metadata = get_image_metadata(
        str(path)
    )

    # Open image
    image = Image.open(path)

    prompt = """
You are the visual evidence analysis component
of TrustLoop, an e-commerce return-abuse triage
system.

Analyze the supplied product-return image.

Do NOT decide whether the customer is fraudulent.
Do NOT make a final refund decision.

Only analyze visible evidence.

Return ONLY valid JSON with exactly these fields:

{
  "image_quality": "GOOD | ACCEPTABLE | POOR",
  "product_condition": "NEW | USED | DAMAGED | UNKNOWN",
  "damage_detected": true,
  "packaging_condition": "INTACT | DAMAGED | MISSING | UNKNOWN",
  "evidence_consistent": true,
  "confidence": 0.0,
  "explanation": "short factual explanation"
}

Rules:

1. Base the answer only on visible evidence.
2. If something cannot be determined from the image,
   use UNKNOWN or null.
3. Do not invent damage or product details.
4. confidence must be between 0 and 1.
5. Keep explanation concise.
6. Do not make a fraud accusation.
"""

    if return_reason:

        prompt += (
            "\n\nCustomer return reason:\n"
            f"{return_reason}\n"
            "\nAssess whether the visible evidence "
            "appears consistent with this stated reason."
        )

    print(
        "\nSending image to Gemini..."
    )

    # ========================================================
    # GEMINI CHAT
    # ========================================================

    chat = client.chats.create(
        model="gemini-3.6-flash"
    )

    # IMPORTANT:
    # Your installed google-genai SDK expects the positional
    # message argument here, not contents= or message=.

    response = chat.send_message(
        [
            image,
            prompt,
        ]
    )

    # ========================================================
    # EXTRACT RESPONSE
    # ========================================================

    raw_text = (response.text or "").strip()


    # Remove markdown code fences if Gemini adds them.

    if raw_text.startswith("```json"):

        raw_text = raw_text[7:]

    elif raw_text.startswith("```"):

        raw_text = raw_text[3:]

    if raw_text.endswith("```"):

        raw_text = raw_text[:-3]

    raw_text = raw_text.strip()

    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        analysis = json.loads(
            raw_text
        )

    except json.JSONDecodeError as exc:

        raise ValueError(
            "Gemini returned invalid JSON.\n\n"
            f"Raw response:\n{raw_text}"
        ) from exc

    # ========================================================
    # REQUIRED FIELDS
    # ========================================================

    required_fields = [
        "image_quality",
        "product_condition",
        "damage_detected",
        "packaging_condition",
        "evidence_consistent",
        "confidence",
        "explanation",
    ]

    missing = [
        field
        for field in required_fields
        if field not in analysis
    ]

    if missing:

        raise ValueError(
            "Gemini response is missing "
            f"fields: {missing}"
        )

    # ========================================================
    # CONFIDENCE VALIDATION
    # ========================================================

    try:

        confidence = float(
            analysis["confidence"]
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise ValueError(
            "Invalid confidence value."
        ) from exc

    if not 0 <= confidence <= 1:

        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    analysis["confidence"] = confidence

    # ========================================================
    # ADD TRUSTLOOP METADATA
    # ========================================================

    analysis["image_path"] = str(
        path
    )

    analysis["metadata"] = metadata

    analysis["model"] = (
        "gemini-3.6-flash"
    )

    analysis["model_status"] = (
        "VISION_ANALYSIS_COMPLETED"
    )

    return analysis


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TRUSTLOOP — GEMINI VISION MODULE")
    print("=" * 70)

    print(
        "\nGemini API key:",
        "CONFIGURED"
        if API_KEY
        else "NOT CONFIGURED"
    )

    print(
        "\nVision module ready."
    )

    print(
        "\nActual image analysis is performed "
        "through analyze_image()."
    )

    print(
        "\n" + "=" * 70
    )