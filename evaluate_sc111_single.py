"""
Skrip Evaluasi Kasus Tunggal WCAG SC 1.1.1 (Ekstraktor Berbasis Prompt LLM)
Berdasarkan Metodologi Artikel:
"Turning manual web accessibility success criteria into automatic: an LLM-based approach"
(Pereira et al., Universal Access in the Information Society, 2024/2025)
Dan Repositori & Model Resmi Penulis:
- Footnote 18: google/pix2struct-textcaps-base (Model VLM Pix2Struct)
- Footnote 19: Salesforce/blip-image-captioning-large (Model VLM BLIP Large)
- Footnote 20: https://gist.github.com/juananpe/98130f0b8f67edbd03f19f498dc10891 (VLM Image Captioning: get-image-description.py)
- Footnote 21: https://gist.github.com/juananpe/34bb3c04b0afb50e1e7b702e5a8c1c5f (LangChain Semantic Similarity: check-1-1-1.py)

Karakteristik Skrip:
1. Mengevaluasi 1 berkas kasus uji HTML secara mandiri dengan alur step-by-step yang transparan.
2. Menggunakan OpenAI(model="gpt-5.6-terra", temperature=0) via openai_patch (pengganti resmi gpt-3.5-turbo-instruct yang telah dideprecate).
3. Tahap 1: Ekstraksi gambar dan teks alternatif HTML dilakukan oleh LLM melalui prompt (Base Prompt dan Ad-hoc Prompt).
4. Tahap 2: Ekstraksi deskripsi visual via VLM (BLIP / Pix2Struct) dan Penilaian Kemiripan Semantik via LangChain.
5. Tahap 3: Menampilkan keputusan akhir: PASSED, FAILED, atau FAILED*.
"""

import os
import sys
import re
import io
import json
import warnings
from PIL import Image
import torch
import pymupdf
from transformers import (
    Pix2StructForConditionalGeneration, Pix2StructProcessor,
    BlipForConditionalGeneration, BlipProcessor
)
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from openai_patch import PatchedOpenAI as OpenAI
from langchain.output_parsers.json import parse_json_markdown

warnings.filterwarnings("ignore")

# Pemuatan variabel lingkungan .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

# Pemilihan backend komputasi untuk model VLM
device = "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# KONFIGURASI EVALUASI KASUS TUNGGAL
# ==============================================================================
# 1. Path berkas HTML kasus uji yang ingin dievaluasi
HTML_FILE_PATH = "testcases/sc111/selected/8c29bcb24ac0f448846a2ffdad4c9693d5aef8c6.html"

# 2. Mode Ad-hoc:
#    - False: Base Prompt LLM hanya mencari tag <img> / <input> standar.
#             (Kasus non-standar CSS bg / <object> akan menghasilkan FAILED*).
#    - True : Mode modifikasi ad-hoc prompt (Seksi 5 Hlm. 12-13 artikel ilmiah).
USE_ADHOC = False

# Ambang batas kemiripan semantik untuk lulus (Seksi 3.3.1 artikel: 0.50)
THRESHOLD = 0.50

# Direktori aset gambar lokal
ASSETS_DIR = os.path.join(BASE_DIR, "test-assets")


# ==============================================================================
# TEMPLATE PROMPT LLM GLOBAL
# ==============================================================================

# 1. Prompt Ekstraksi Gambar & Teks Alternatif (Mendukung Sisipan Notice Ad-hoc) (Sesuai Seksi 3.3.1 & 5 Artikel)
EXTRACTOR_TEMPLATE = """Given the following HTML code:

```html
{html_code}
```

Extract image src and alternative text from the tag img or input element.
{special_instruction}

Write your answer in a JSON object with two keys: image_src (string or null) and alt_text (string or null).
"""

# 2. Semantic Similarity Prompt (Footnote 21)
SIMILARITY_TEMPLATE = """Given the following true sentence:
 
    {true_sentence}

Tell me how similar is that sentence to the following one:

	{sentence}


Write your answer in a JSON object with one key: similarity (a float from 0 to 1) 
"""


# ==============================================================================
# VARIABEL GLOBAL MODEL & CHAINS (LAZY LOADING)
# ==============================================================================
EXTRACTOR_CHAIN = None
SIM_CHAIN = None

PIX_MODEL = None
PIX_PROC = None
BLIP_MODEL = None
BLIP_PROC = None


# ==============================================================================
# INISIALISASI SELURUH RANTAI PROMPT LLM (EKSTRAKTOR & SIMILARITY)
# ==============================================================================
def init_llm_chains():
    """Membangun dan memuat seluruh rantai LLM ke variabel global sekali saja jika belum dimuat."""
    global EXTRACTOR_CHAIN, SIM_CHAIN

    if EXTRACTOR_CHAIN is None or SIM_CHAIN is None:
        model_name = "gpt-5.6-terra"
        print("\n[INISIALISASI LLM OPENAI VIA LANGCHAIN]")
        print("-" * 80)
        print(f"Menghubungkan OpenAI {model_name} (temperature=0)...")
        llm = OpenAI(model_name=model_name, temperature=0)

        extractor_prompt = PromptTemplate(
            template=EXTRACTOR_TEMPLATE,
            input_variables=['html_code', 'special_instruction']
        )
        EXTRACTOR_CHAIN = LLMChain(prompt=extractor_prompt, llm=llm)

        sim_prompt = PromptTemplate(
            template=SIMILARITY_TEMPLATE,
            input_variables=['true_sentence', 'sentence']
        )
        SIM_CHAIN = LLMChain(prompt=sim_prompt, llm=llm)

    return EXTRACTOR_CHAIN, SIM_CHAIN


# ==============================================================================
# INISIALISASI VLM (FOOTNOTE 18, 19, & 20)
# ==============================================================================
def init_vlm_models():
    """Memuat model VLM Google Pix2Struct dan Salesforce BLIP ke variabel global sekali saja jika belum dimuat."""
    global PIX_MODEL, PIX_PROC, BLIP_MODEL, BLIP_PROC

    if PIX_MODEL is None or BLIP_MODEL is None:
        print("\n[INISIALISASI VLM]")
        print("-" * 80)
        print("[1/2] Memuat Google Pix2Struct (google/pix2struct-textcaps-base)...")
        PIX_MODEL = Pix2StructForConditionalGeneration.from_pretrained('google/pix2struct-textcaps-base').to(device)
        PIX_PROC = Pix2StructProcessor.from_pretrained('google/pix2struct-textcaps-base')

        print("[2/2] Memuat Salesforce BLIP Large (Salesforce/blip-image-captioning-large)...")
        BLIP_MODEL = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-large').to(device)
        BLIP_PROC = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-large')

    return PIX_MODEL, PIX_PROC, BLIP_MODEL, BLIP_PROC


# ==============================================================================
# PARSING OUTPUT JSON EKSTRAKTOR LLM
# ==============================================================================
def parse_extractor_json(response_str: str):
    """Mem-parsing output JSON dari LLM menggunakan parser bawaan LangChain."""
    try:
        data = parse_json_markdown(response_str)
        if isinstance(data, list):
            return data[0] if (data and isinstance(data[0], dict)) else {}
        elif isinstance(data, dict):
            return data
        return {}
    except Exception:
        # Fallback regex jika ada teks pembuka tak terduga
        img_m = re.search(r'"image_src":\s*(?:"([^"]+)"|null)', response_str)
        alt_m = re.search(r'"alt_text":\s*(?:"([^"]+)"|null)', response_str)
        return {
            "image_src": img_m.group(1) if (img_m and img_m.group(1)) else None,
            "alt_text": alt_m.group(1) if (alt_m and alt_m.group(1)) else None
        }


# ==============================================================================
# PEMUATAN GAMBAR DARI TEST-ASSETS & RASTERISASI SVG
# ==============================================================================
def load_image(image_src: str):
    """
    Memuat gambar langsung dari test-assets/. Jika format SVG, rasterisasi in-memory via PyMuPDF.
    """
    if not image_src:
        return None, "URL berkas gambar kosong"

    fname = os.path.basename(image_src)
    asset_path = os.path.join(ASSETS_DIR, fname)

    if not os.path.exists(asset_path):
        return None, f"Berkas aset '{fname}' tidak ditemukan di folder test-assets/"

    with open(asset_path, "rb") as f:
        raw_bytes = f.read()

    if fname.lower().endswith(".svg") or b"<svg" in raw_bytes[:100]:
        doc = pymupdf.open(stream=raw_bytes, filetype="svg")
        pix = doc[0].get_pixmap(alpha=True, matrix=pymupdf.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
        return img, f"Format SVG -> Berhasil dirasterisasi ke RGB ({img.width}x{img.height} px)"
    else:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        return img, f"Format Raster -> Berhasil dimuat ({img.width}x{img.height} px)"


# ==============================================================================
# EKSTRAKSI DESKRIPSI VISUAL OLEH MODEL VLM
# ==============================================================================
def generate_vlm_caption(image: Image.Image, is_icon: bool):
    """Ekstraksi deskripsi visual menggunakan model VLM global yang telah dimuat."""
    global PIX_MODEL, PIX_PROC, BLIP_MODEL, BLIP_PROC

    if PIX_MODEL is None or BLIP_MODEL is None:
        init_vlm_models()

    if is_icon:
        inputs = BLIP_PROC(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = BLIP_MODEL.generate(**inputs, max_length=60)
        caption = BLIP_PROC.decode(output_ids[0], skip_special_tokens=True)
        used_model = "Salesforce BLIP Large"
    else:
        inputs = PIX_PROC(images=image, return_tensors='pt').to(device)
        with torch.no_grad():
            output_ids = PIX_MODEL.generate(**inputs, max_length=60)
        caption = PIX_PROC.decode(output_ids[0], skip_special_tokens=True)
        used_model = "Google Pix2Struct Base"

    return caption.strip(), used_model


# ==============================================================================
# PIPELINE EKSEKUSI UTAMA EVALUASI TUNGGAL BERBASIS LLM
# ==============================================================================
def run_single_evaluation(file_path: str = HTML_FILE_PATH, use_adhoc: bool = USE_ADHOC):
    print("=" * 80)
    print("EVALUASI KASUS TUNGGAL WCAG SC 1.1.1 (EKSTRAKTOR BERBASIS PROMPT LLM)")
    print("=" * 80)
    print(f"Target File : {file_path}")
    print(f"Mode Ad-hoc : {'AKTIF (Modifikasi Prompt Ad-hoc)' if use_adhoc else 'NON-AKTIF'}")
    print(f"Device VLM  : {device}")

    # Inisialisasi objek hasil evaluasi
    result = {
        "prompt_type": None,
        "ext_status": None,
        "img_src": None,
        "alt_text": None,
        "vlm_model": None,
        "vlm_caption": None,
        "sim_score": None,
        "eval_status": None
    }

    # 1. Validasi Keberadaan Berkas HTML
    if not os.path.exists(file_path):
        print(f"\n[ERROR] Berkas '{file_path}' tidak ditemukan!")
        result["eval_status"] = "ERROR"
        return result

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Inisialisasi Model LLM OpenAI untuk Ekstraksi dan Kemiripan Semantik
    extractor_chain, sim_chain = init_llm_chains()

    # 3. Inisialisasi Model VLM
    init_vlm_models()

    # --------------------------------------------------------------------------
    # [TAHAP 1] Ekstraksi Elemen HTML Menggunakan LLM Prompt
    # --------------------------------------------------------------------------
    print("\nEKSTRAKSI ELEMEN HTML BERBASIS PROMPT LLM")
    print("-" * 80)
    print("Kode HTML Kasus Uji:")
    for line in html_content.strip().splitlines():
        print(f"  | {line}")

    # Penentuan Petunjuk Khusus / Special Instruction
    if not use_adhoc:
        prompt_type = "Base LLM Prompt"
        special_instruction = ""
    else:
        # Sisipan notice khusus berdasarkan konten HTML (Seksi 5 Hlm. 12-13 Artikel)
        if "background-image" in html_content:
            prompt_type = "Ad-hoc Prompt (CSS background-image)"
            special_instruction = "Notice that the image is obtained using the CSS background-image property.\n"
        elif "<object" in html_content:
            prompt_type = "Ad-hoc Prompt (Tag <object> aria-labelledby)"
            special_instruction = "Notice that the image is obtained using the aria-labelledby attribute associating the <object> element with the <span> element.\n"
        else:
            prompt_type = "Standard Prompt"
            special_instruction = ""

    raw_extractor_output = extractor_chain.run({
        'html_code': html_content,
        'special_instruction': special_instruction
    })

    print("\nRespon Mentah LLM:")
    print(raw_extractor_output.strip())
    parsed = parse_extractor_json(raw_extractor_output)

    img_src = parsed.get("image_src")
    alt_text = parsed.get("alt_text")
    ext_status = "FOUND" if img_src else "NOT FOUND"

    result.update({
        "prompt_type": prompt_type,
        "ext_status": ext_status,
        "img_src": img_src,
        "alt_text": alt_text
    })

    print(f"\nHasil Parsing Ekstraksi:")
    print(f"  -> Tipe Prompt   : {prompt_type}")
    print(f"  -> Status        : {ext_status}")
    print(f"  -> Image Source  : {img_src if img_src else '-'}")
    print(f"  -> Alt Text      : {alt_text if alt_text else '-'}")

    # Jika ekstraksi gagal pada Base Prompt
    if not img_src:
        print("\nKEPUTUSAN AKHIR EVALUASI")
        print("-" * 80)
        print("Status Evaluasi : FAILED*")
        print("Penjelasan      : Elemen gambar tidak ditemukan oleh Base Prompt karena tidak")
        print("                  menggunakan tag <img> atau <input> standar (misalnya kasus")
        print("                  non-standar CSS background atau tag <object>).")
        print("                  Aktifkan 'USE_ADHOC = True' untuk menggunakan prompt modifikasi.")
        print("=" * 80)
        result["eval_status"] = "FAILED*"
        return result

    # --------------------------------------------------------------------------
    # [TAHAP 2] Pemuatan Berkas Gambar Lokal & Rasterisasi
    # --------------------------------------------------------------------------
    print("\nPEMUATAN GAMBAR DARI test-assets/")
    print("-" * 80)
    img_obj, load_msg = load_image(img_src)
    print(f"Status Pemuatan : {load_msg}")

    if not img_obj:
        print("\n[ERROR] Gambar gagal diproses. Evaluasi dihentikan.")
        result["eval_status"] = "ERROR"
        return result

    # --------------------------------------------------------------------------
    # [TAHAP 3] Ekstraksi Deskripsi Visual Gambar oleh VLM
    # --------------------------------------------------------------------------
    print("\nDESKRIPSI VISUAL GAMBAR OLEH VLM")
    print("-" * 80)
    is_icon = img_src.lower().endswith(".svg")
    caption, used_vlm = generate_vlm_caption(img_obj, is_icon)
    result["vlm_model"] = used_vlm
    result["vlm_caption"] = caption

    print(f"Model Digunakan : {used_vlm}")
    print(f"Hasil Caption   : \"{caption}\"")

    # --------------------------------------------------------------------------
    # [TAHAP 4] Perhitungan Kemiripan Semantik
    # --------------------------------------------------------------------------
    print("\nPERHITUNGAN KEMIRIPAN SEMANTIK")
    print("-" * 80)
    print(f"Kalimat Acuan (True Sentence)       : \"{caption}\"")
    print(f"Kalimat Uji (Candidate Sentence)    : \"{alt_text}\"")

    raw_sim_output = sim_chain.run({'true_sentence': caption, 'sentence': alt_text})
    try:
        sim_data = json.loads(raw_sim_output.strip())
        sim_score = float(sim_data.get('similarity', 0.0))
    except Exception:
        match = re.search(r'\"similarity\":\s*([0-9\.]+)', raw_sim_output)
        sim_score = float(match.group(1)) if match else 0.0

    print(f"Respon JSON Mentah                  : {raw_sim_output.strip()}")
    print(f"Skor Kemiripan Semantik (0.0 - 1.0) : {sim_score:.2f}")

    # --------------------------------------------------------------------------
    # [TAHAP 5] Klasifikasi Keputusan Kepatuhan WCAG SC 1.1.1
    # --------------------------------------------------------------------------
    print("\nKEPUTUSAN AKHIR EVALUASI")
    print("-" * 80)
    is_passed = sim_score >= THRESHOLD
    eval_status = "PASSED" if is_passed else "FAILED"

    result["sim_score"] = sim_score
    result["eval_status"] = eval_status

    print(f"Status Evaluasi : {eval_status}")
    print(f"Ambang Batas    : Skor >= {THRESHOLD:.2f} untuk PASSED")
    print(f"Skor Diperoleh  : {sim_score:.2f}")

    if is_passed:
        print("Kesimpulan      : Teks alternatif yang diekstrak oleh LLM SESUAI secara makna")
        print("                  dengan deskripsi visual gambar dari model VLM (Memenuhi WCAG SC 1.1.1).")
    else:
        print("Kesimpulan      : Teks alternatif yang diekstrak oleh LLM TIDAK SESUAI secara makna")
        print("                  dengan deskripsi visual gambar (Melanggar WCAG SC 1.1.1).")
    print("=" * 80)

    return result


if __name__ == "__main__":
    # Penggunaan via Terminal:
    # python evaluate_sc111_single.py testcases/sc111/selected/8c29bcb24ac0f448846a2ffdad4c9693d5aef8c6.html --adhoc
    target_path = HTML_FILE_PATH
    adhoc_flag = USE_ADHOC

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    if "--adhoc" in sys.argv:
        adhoc_flag = True

    run_single_evaluation(file_path=target_path, use_adhoc=adhoc_flag)
