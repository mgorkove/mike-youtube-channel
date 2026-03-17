"""Generate background images for revenge_chronicles channel."""

import io
import os
from pathlib import Path

from google import genai
from google.genai import types
import numpy as np
from PIL import Image, ImageFilter

from dotenv import load_dotenv
load_dotenv()

OUTPUT_DIR = Path("channels/revenge_chronicles/assets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash-image"


def generate_image(prompt: str, width: int, height: int, output_path: Path):
    """Generate a single image with Gemini and save it."""
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    for part in response.candidates[0].content.parts:
        if part.inline_data and part.inline_data.mime_type and \
           part.inline_data.mime_type.startswith("image/"):
            img = Image.open(io.BytesIO(part.inline_data.data))
            img = img.resize((width, height), Image.LANCZOS)
            img.save(output_path, "PNG")
            print(f"Saved: {output_path} ({width}x{height})")
            return img

    raise RuntimeError("No image data in response")


# --- Landscape background (1280x704) ---
print("Generating landscape background...")
bg_prompt = (
    "A photorealistic moody scene of a rain-streaked floor-to-ceiling window at night, "
    "looking out at a blurred city skyline with warm golden and cool blue lights. Inside, "
    "a cozy window seat with velvet cushions, a throw blanket, and a half-empty cup of tea. "
    "Dim warm lamplight from one side. Raindrops visible on the glass. Dark, cinematic, "
    "atmospheric. No people. No text. Wide angle, 16:9 composition."
)
bg_img = generate_image(bg_prompt, 1280, 704, OUTPUT_DIR / "background_nobutton.png")

# Copy subscribe button from heartbreak_chronicles using a precise pixel mask
print("Copying subscribe button from heartbreak_chronicles...")
hb_bg = Image.open("channels/heartbreak_chronicles/assets/background.png")
hb_arr = np.array(hb_bg)

# Build mask: for each row, fill from leftmost to rightmost red pixel
# This traces the exact shape of the red subscribe button
red = (hb_arr[:,:,0] > 180) & (hb_arr[:,:,1] < 60) & (hb_arr[:,:,2] < 60)
mask = np.zeros(hb_arr.shape[:2], dtype=np.uint8)
for row in range(0, 130):
    row_red = np.where(red[row, :420])[0]
    if len(row_red) > 0:
        mask[row, row_red[0]:row_red[-1]+1] = 255
mask_img = Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=1))

bg_img.paste(hb_bg, (0, 0), mask_img)
bg_img.save(OUTPUT_DIR / "background.png", "PNG")
print(f"Saved: {OUTPUT_DIR / 'background.png'} (1280x704 with subscribe button)")

# Clean up
(OUTPUT_DIR / "background_nobutton.png").unlink(missing_ok=True)

# --- Shorts background (1024x1024) ---
print("Generating shorts background...")
shorts_prompt = (
    "A photorealistic portrait of a confident beautiful woman in her mid-20s with long dark hair, "
    "wearing a sleek dark blazer over a simple top. She has her arms crossed and is looking directly "
    "at the camera with a calm, knowing expression — like she has a secret. Dark moody background "
    "with soft warm lighting from one side. Cinematic portrait photography. No text. Square composition."
)
generate_image(shorts_prompt, 1024, 1024, OUTPUT_DIR / "background_shorts.png")

print("\nDone! Backgrounds saved to channels/revenge_chronicles/assets/")
