"""Optional manual check against the real Groq API.

This file is not collected by pytest because its name does not start with
"test_". It makes one real API request, so run it only when you want to test
your key and model.

Usage:
    python -m tests.manual_image_qa --image path/to/image.png
    python -m tests.manual_image_qa --image diagram.png --question "Explain this diagram."
"""

import argparse
from pathlib import Path

from modules.ai_assistant import AIAssistantError, ConfigurationError, ask_image
from modules.image_processor import ImageValidationError, prepare_image


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual Groq image question test")
    parser.add_argument("--image", required=True, help="Path to a JPEG, PNG, or WEBP image")
    parser.add_argument(
        "--question",
        default="What is in this image?",
        help="Question to ask about the image",
    )
    args = parser.parse_args()

    try:
        original_bytes = Path(args.image).read_bytes()
        prepared_bytes, mime_type = prepare_image(original_bytes)
        answer = ask_image(args.question, prepared_bytes, mime_type)
    except (OSError, ConfigurationError, ImageValidationError, AIAssistantError) as error:
        print(f"Error: {error}")
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
