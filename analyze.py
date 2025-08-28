import anthropic
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def _pil_mime(img: Image.Image) -> str:
    fmt = (img.format or "PNG").upper()
    mapping = {
        "PNG": "image/png",
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "WEBP": "image/webp",
        "GIF": "image/gif"
    }
    return mapping.get(fmt, "image/png")

def get_llm_response(image_data: bytes) -> str:
    # references: https://docs.anthropic.com/en/docs/build-with-claude/vision
    # https://docs.anthropic.com/en/api/messages-examples
    image = Image.open(io.BytesIO(image_data))
    mime = _pil_mime(image)

    b64 = base64.standard_b64encode(image_data).decode("utf-8")
    prompt = (
        "You are given a single image. "
        "1) Provide a concise caption (<= 15 words). "
        "2) List the top 3 object categories seen with approximate counts. "
        "Return JSON only with keys: caption (string), objects (array of {label (string), count (int)}). "
    )

    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=400,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                ],
            }
        ],
    )

    text_blocks = [c for c in resp.content if c.type == "text"]
    text = text_blocks[0].text.strip() if text_blocks else ""
    return text or '{"caption":"(no result)","objects":[]}'
    