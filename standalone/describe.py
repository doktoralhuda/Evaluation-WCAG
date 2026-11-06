import sys
import io
import warnings

if len(sys.argv) < 2:
    print("Penggunaan: python describe.py <path_gambar>")
    sys.exit(1)

from PIL import Image
import torch
import pymupdf
from transformers import (
    Pix2StructForConditionalGeneration, Pix2StructProcessor,
    BlipForConditionalGeneration, BlipProcessor
)

warnings.filterwarnings("ignore")

image_path = sys.argv[1]
device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# Load gambar (rasterisasi jika SVG)
with open(image_path, "rb") as f:
    raw_bytes = f.read()

if image_path.lower().endswith(".svg") or b"<svg" in raw_bytes[:100]:
    doc = pymupdf.open(stream=raw_bytes, filetype="svg")
    pix = doc[0].get_pixmap(alpha=True, matrix=pymupdf.Matrix(2, 2))
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    is_icon = True
else:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    is_icon = False

# Generate deskripsi gambar via VLM
if is_icon:
    model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large').to(device)
    processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')
else:
    model = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-textcaps-base').to(device)
    processor = Pix2StructProcessor.from_pretrained('google/pix2struct-textcaps-base')

inputs = processor(images=img, return_tensors='pt').to(device)
with torch.no_grad():
    output_ids = model.generate(**inputs, max_length=60)

caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()

print(f"Deskripsi Gambar: {caption}")

