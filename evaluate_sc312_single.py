"""
Skrip Evaluasi Kasus Tunggal WCAG SC 3.1.2 (Language of Parts Berbasis Prompt LLM)
Berdasarkan Metodologi Artikel:
"Turning manual web accessibility success criteria into automatic: an LLM-based approach"
(Pereira et al., Universal Access in the Information Society, 2024/2025), Seksi 3.3.3 & Fig. 4.

Karakteristik Skrip:
1. Mengevaluasi 1 berkas kasus uji HTML secara mandiri dengan alur step-by-step yang transparan.
2. Menggunakan ChatOpenAI(model="gpt-5.6-sol", temperature=0) via openai_patch (PatchedChatOpenAI).
3. Tahap 1: Ekstraksi dan evaluasi konsistensi tag bahasa teks HTML oleh LLM via Base Prompt (Fig. 4).
4. Tahap 2: Menampilkan keputusan akhir: PASSED, FAILED, atau FAILED*.
"""

import os
import sys
import re
import json
import warnings
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from openai_patch import PatchedChatOpenAI as ChatOpenAI
from langchain.output_parsers.json import parse_json_markdown

warnings.filterwarnings("ignore")

# Pemuatan variabel lingkungan .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ==============================================================================
# KONFIGURASI EVALUASI KASUS TUNGGAL
# ==============================================================================
# 1. Path berkas HTML kasus uji yang ingin dievaluasi (default)
HTML_FILE_PATH = "testcases/sc312/selected/ec40c0a032b11cabc03d71b6884ab9b85ee160ad.html"

# 2. Mode Ad-hoc:
#    - Di paper Halaman 13 dinyatakan SC 3.1.2 tidak memerlukan modifikasi ad-hoc prompt khusus.
USE_ADHOC = False


# ==============================================================================
# TEMPLATE PROMPT LLM GLOBAL (Sesuai Seksi 3.3.3 Hlm. 8-9 & Fig. 4 Artikel Ilmiah)
# ==============================================================================
# LANGUAGE_CHECK_TEMPLATE = """Based on the following HTML code, generate a JSON object representing the language identification of each text and if that identification is correct or not:

# ```{html_code}```

# Guidelines:
# 1. Evaluate each text passage within its scope (for mixed-language elements, evaluate each text segment).
# 2. Bilingual homographs (phrases valid or meaningful in multiple languages) are correct for either language.

# Don't include any further explanation, just the JSON object
# """

LANGUAGE_CHECK_TEMPLATE = """Based on the following HTML code, generate a JSON object representing the language identification of each text and if that identification is correct or not:

```{html_code}```

Don't include any further explanation, just the JSON object
"""


# ==============================================================================
# PARSING OUTPUT JSON DARI LLM
# ==============================================================================
def parse_language_eval_json(raw_output: str):
    """
    Mem-parsing respon JSON dari LLM untuk ekstraksi status kecocokan bahasa.
    Menangani variasi kunci (isCorrect, correct, status, lang, text) dan hierarki objek.
    """
    try:
        data = parse_json_markdown(raw_output)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", raw_output, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except Exception:
            return []

    results = []

    def extract_from_obj(obj, tag_name=""):
        if isinstance(obj, dict):
            corr_val, lang_val, text_val = None, "-", tag_name or "-"

            # Ekstraksi field dari pasangan key-value
            for k, v in obj.items():
                k_lower = k.lower()

                # 1. Deteksi status kebenaran (boolean atau string status)
                if corr_val is None:
                    if any(term in k_lower for term in ["correct", "valid", "match", "status", "result", "ident"]):
                        if isinstance(v, bool):
                            corr_val = v
                        elif isinstance(v, (int, float)):
                            corr_val = bool(v)
                        elif isinstance(v, str):
                            v_lower = v.lower().strip()
                            if v_lower in ["true", "correct", "valid", "yes", "match", "passed", "1"]:
                                corr_val = True
                            elif v_lower in ["false", "incorrect", "invalid", "no", "failed", "mismatch", "0"]:
                                corr_val = False
                    elif isinstance(v, str):
                        v_lower = v.lower().strip()
                        if v_lower in ["correct", "valid", "match"]:
                            corr_val = True
                        elif v_lower in ["incorrect", "invalid", "mismatch"]:
                            corr_val = False

                # 2. Deteksi representasi bahasa (lang, locale, code)
                if any(term in k_lower for term in ["lang", "code", "locale"]) and isinstance(v, (str, int)):
                    lang_val = str(v)

                # 3. Deteksi teks/konten
                if any(term in k_lower for term in ["text", "content", "phrase", "snippet", "excerpt"]) and isinstance(v, str):
                    text_val = str(v)

            # Simpan objek jika memiliki status evaluasi
            if corr_val is not None:
                results.append({
                    "element": tag_name if tag_name else "text",
                    "text": text_val,
                    "language": lang_val,
                    "is_correct": corr_val
                })

            # Rekursi ke child dict/list jika ada struktur bersarang
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract_from_obj(v, tag_name=str(k))

        elif isinstance(obj, list):
            for item in obj:
                extract_from_obj(item, tag_name)

    extract_from_obj(data)
    return results


# ==============================================================================
# INISIALISASI RANTAI PROMPT LLM (GPT-4)
# ==============================================================================
LANGUAGE_EVAL_CHAIN = None

def init_language_eval_chain():
    """Memuat rantai LLM ChatOpenAI (gpt-5.6-sol via openai_patch) jika belum dimuat."""
    global LANGUAGE_EVAL_CHAIN

    if LANGUAGE_EVAL_CHAIN is None:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("Kunci API tidak ditemukan! Harap sediakan OPENAI_API_KEY di .env.")

        model_name = "gpt-5.6-sol"
        print("\n[INISIALISASI LLM OPENAI VIA LANGCHAIN]")
        print("-" * 80)
        print(f"Menghubungkan ChatOpenAI {model_name} (temperature=0)...")
        llm = ChatOpenAI(model_name=model_name, temperature=0)

        prompt = PromptTemplate(
            template=LANGUAGE_CHECK_TEMPLATE,
            input_variables=["html_code"]
        )
        LANGUAGE_EVAL_CHAIN = LLMChain(prompt=prompt, llm=llm)

    return LANGUAGE_EVAL_CHAIN


# ==============================================================================
# PIPELINE EKSEKUSI UTAMA EVALUASI TUNGGAL SC 3.1.2
# ==============================================================================
def run_single_evaluation(file_path: str = HTML_FILE_PATH, use_adhoc: bool = USE_ADHOC):
    """
    Menjalankan evaluasi kasus tunggal untuk WCAG SC 3.1.2.
    Mengembalikan dictionary hasil evaluasi yang kompatibel dengan evaluate.py.
    """
    print("=" * 80)
    print("EVALUASI KASUS TUNGGAL WCAG SC 3.1.2 (LANGUAGE OF PARTS BERBASIS LLM)")
    print("=" * 80)
    print(f"Target File : {file_path}")
    print(f"Mode Ad-hoc : {'AKTIF' if use_adhoc else 'NON-AKTIF'}")

    result = {
        "prompt_type": "Ad-hoc Prompt" if use_adhoc else "Base LLM Prompt",
        "ext_status": None,
        "items": [],
        "score": None,
        "eval_status": None
    }

    # 1. Validasi Keberadaan Berkas HTML Kasus Uji
    if not os.path.exists(file_path):
        print(f"\n[ERROR] Berkas '{file_path}' tidak ditemukan!")
        result["eval_status"] = "ERROR"
        return result

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Inisialisasi Rantai LLM LangChain (ChatOpenAI gpt-5.6-sol via openai_patch)
    chain = init_language_eval_chain()

    # --------------------------------------------------------------------------
    # [TAHAP 1] Pemeriksaan Konsistensi Tag Bahasa via LLM (Fig. 4 Artikel Ilmiah)
    # --------------------------------------------------------------------------
    print("\nPEMERIKSAAN KONSISTENSI BAHASA BAGIAN VIA LLM")
    print("-" * 80)
    print("Kode HTML Kasus Uji:")
    for line in html_content.strip().splitlines():
        print(f"  | {line}")

    print("\nMenjalankan evaluasi konsistensi tag bahasa dengan LLM...")
    raw_output = chain.run(html_code=html_content)

    print("\nRespon Mentah LLM:")
    print(raw_output.strip())

    items = parse_language_eval_json(raw_output)
    result["items"] = items

    ext_status = "FOUND" if items else "NOT FOUND"
    result["ext_status"] = ext_status

    if not items:
        print("\nKEPUTUSAN AKHIR EVALUASI")
        print("-" * 80)
        print("Status Evaluasi : FAILED*")
        print("Penjelasan      : Gagal mengekstrak struktur verifikasi bahasa dari respon LLM.")
        print("=" * 80)
        result["eval_status"] = "FAILED*"
        result["score"] = None
        return result

    print("\nHasil Parsing Identifikasi Bahasa:")
    all_correct = True
    for idx, item in enumerate(items, 1):
        status_label = "VALID" if item["is_correct"] else "TIDAK VALID"
        print(f"  {idx}. Elemen : <{item['element']}> | Bahasa: '{item['language']}' | Status: {status_label}")
        if item["text"] != "-":
            print(f"     Teks   : \"{item['text'][:60]}{'...' if len(item['text']) > 60 else ''}\"")
        if not item["is_correct"]:
            all_correct = False

    # --------------------------------------------------------------------------
    # [TAHAP 2] Klasifikasi Keputusan Kepatuhan WCAG SC 3.1.2
    # Berdasarkan Metodologi Artikel Ilmiah: Seksi 3.3.3 Hlm. 8-9 & Fig. 4
    # (Pereira et al., Universal Access in the Information Society, 2024/2025)
    #
    # Aturan Penentuan Keputusan (Passed / Failed):
    # - PASSED: Seluruh teks dievaluasi benar oleh LLM (semua is_correct = True).
    # - FAILED: Ditemukan setidaknya satu kesalahan identifikasi bahasa (is_correct = False).
    # --------------------------------------------------------------------------
    print("\nKEPUTUSAN AKHIR EVALUASI")
    print("-" * 80)
    eval_status = "PASSED" if all_correct else "FAILED"
    score_val = 1.0 if all_correct else 0.0

    result["eval_status"] = eval_status
    result["score"] = score_val

    print(f"Status Evaluasi : {eval_status}")
    print(f"Skor Kesesuaian : {score_val:.2f}")

    if all_correct:
        print("Kesimpulan      : Seluruh penandaan bahasa (lang) pada bagian konten valid dan sesuai")
        print("                  dengan bahasa teks sebenarnya (Memenuhi kriteria WCAG SC 3.1.2).")
    else:
        print("Kesimpulan      : Ditemukan inkonsistensi atau kesalahan tag bahasa pada teks")
        print("                  (Melanggar kriteria WCAG SC 3.1.2).")
    print("=" * 80)

    return result


if __name__ == "__main__":
    # Penggunaan via Terminal:
    # python evaluate_sc312_single.py testcases/sc312/selected/ec40c0a032b11cabc03d71b6884ab9b85ee160ad.html
    # python evaluate_sc312_single.py testcases/sc312/selected/ec40c0a032b11cabc03d71b6884ab9b85ee160ad.html --adhoc
    target_path = HTML_FILE_PATH
    adhoc_flag = USE_ADHOC

    if len(sys.argv) > 1 and not sys.argv[1].startswith("--"):
        target_path = sys.argv[1]
    if "--adhoc" in sys.argv:
        adhoc_flag = True

    run_single_evaluation(file_path=target_path, use_adhoc=adhoc_flag)
