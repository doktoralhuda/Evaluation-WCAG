"""
Skrip Evaluasi Kasus Tunggal WCAG SC 2.4.4 (Ekstraktor Berbasis Prompt LLM)
Berdasarkan Metodologi Artikel:
"Turning manual web accessibility success criteria into automatic: an LLM-based approach"
(Pereira et al., Universal Access in the Information Society, 2024/2025)
Dan Repositori Gist Resmi Penulis (Footnote 23: https://gist.github.com/juananpe/efbd91cbf0bf79fd93ae86106ede0d91)

Karakteristik Skrip:
1. Mengevaluasi 1 berkas kasus uji HTML secara mandiri dengan alur step-by-step yang transparan.
2. Menggunakan ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0) sesuai kode Gist resmi artikel (Footnote 23).
3. Tahap 1: Ekstraksi tautan HTML dilakukan oleh LLM melalui prompt (Base Prompt dan Ad-hoc Prompt).
4. Tahap 2: Peringkasan & Penilaian Kesesuaian Semantik via LangChain Agent yang menjelajahi halaman web sasaran secara otonom.
5. Tahap 3: Menampilkan keputusan akhir: PASSED, FAILED, atau FAILED*.
"""

import os
import sys
import re
import warnings
import html
import requests
from dotenv import load_dotenv
from langchain import PromptTemplate, LLMChain
from anthropic_patch import PatchedChatAnthropic as ChatAnthropic
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers.json import parse_json_markdown

warnings.filterwarnings("ignore")

# Pemuatan variabel lingkungan .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


# ==============================================================================
# KONFIGURASI EVALUASI KASUS TUNGGAL
# ==============================================================================
# 1. Path berkas HTML kasus uji yang ingin dievaluasi
HTML_FILE_PATH = "testcases/sc244/selected/a8cc66de4d60e34c7ee0d09fd6ab965ac23d9b4f.html"

# 2. Mode Ad-hoc:
#    - False: Base Prompt LLM hanya mencari elemen jangkar <a> standar.
#             (Kasus non-standar seperti ARIA labelledby/label, <area> map, atau button redirect akan berstatus FAILED*).
#    - True : Mode modifikasi ad-hoc prompt (Seksi 5 Hlm. 12-13 artikel ilmiah).
USE_ADHOC = False

# Ambang batas kesesuaian semantik untuk lulus (Seksi 3.3.1 & 3.3.2 artikel: 0.50)
THRESHOLD = 0.50


# ==============================================================================
# TEMPLATE PROMPT LLM GLOBAL
# ==============================================================================

# 1. Prompt Ekstraksi Tautan & Teks Deskripsi (Mendukung Sisipan Notice Ad-hoc) (Sesuai Seksi 3.3.2 & 5 Artikel)
EXTRACTOR_TEMPLATE = """Given the following HTML code:

```html
{html_code}
```

Extract the link destination URL from the <a> tag href attribute and its link description.
Rules for link description:
1. Extract the text directly inside the <a> tag (or alt/title attribute of an enclosed <img>). Do not look up references outside the tag.
2. If the anchor text is only a format or generic term, combine it with its link context. Otherwise, use only the anchor text itself.

{special_instruction}

Write your answer in a single JSON object with two keys: link_url (string or null) and link_text (string or null). If multiple links exist, extract the first one as a single JSON object.
"""

# alternatif:
# 2. If the anchor text is only a format or generic term, combine it with its parent context (heading, list, or table). Otherwise, use only the anchor text itself.

# prompt ekstraktor di bawah dapat menghasilkan akurasi 90%
# EXTRACTOR_TEMPLATE = """Given the following HTML code:

# ```html
# {html_code}
# ```

# Extract the link destination URL from the <a> tag href attribute and its link description.
# Rules for link description:
# 1. Extract the text directly inside the <a> tag (or alt/title of any enclosed <img>).
# 2. Only if the anchor text is merely a generic file format name (e.g. 'HTML', 'EPUB', 'Plain text', 'PDF'), combine it with its parent heading/list/table context (e.g. 'Ulysses - HTML'). For all other descriptive links (e.g. 'Contact us', 'About us', 'Call us', etc.), use the anchor text itself directly without prepending surrounding sentence text.

# {special_instruction}

# Write your answer in a single JSON object with two keys: link_url (string or null) and link_text (string or null). If multiple links exist, extract the first one as a single JSON object.
# """

# 2. Prompt Peringkasan & Penilaian Kesesuaian Semantik untuk LangChain Agent
# (Persis sesuai Seksi 3.3.2 Halaman 8 & Gist Footnote 23: check-2-4-4.py)
SUITABILITY_AGENT_PROMPT = """Browse to {url} and summarize the content in one sentence. Then, tell me on a scale from 0 to 1 how suitable it would be to add a link to that website using this text for the anchor: '{link_text}'. Write your answer as a JSON object with the keys 'summary' and 'suitability'. For example {{"summary": "summary of the page", "suitability": a float number between 0 and 1}}"""

# 3. Prompt Khusus Ad-hoc Evaluasi Tombol Pengalihan (Button Redirection role='link')
# (Persis sesuai kutipan metodologi Seksi 5 Halaman 13 Artikel Ilmiah:
#  "browse to this page ... and tell me if the text that describes where I will go if I click the button is right")
ADHOC_BUTTON_PROMPT = """browse to this page {target_page} and tell me if the text that describes where I will go if I click the button is right"""


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
        # Fallback regex jika ada teks pembuka/penutup tak terduga
        url_m = re.search(r'"link_url":\s*(?:"([^"]+)"|null)', response_str)
        text_m = re.search(r'"link_text":\s*(?:"([^"]+)"|null)', response_str)
        return {
            "link_url": url_m.group(1) if (url_m and url_m.group(1)) else None,
            "link_text": text_m.group(1) if (text_m and text_m.group(1)) else None
        }

# ==============================================================================
# PARSING OUTPUT JSON SUITABILITY AGENT (PERINGKASAN & KESESUAIAN)
# ==============================================================================
def parse_suitability_json(response_str: str):
    """Mem-parsing output JSON peringkasan dan kesesuaian dari LLM/Agent menggunakan parser bawaan LangChain."""
    try:
        data = parse_json_markdown(response_str)
        if isinstance(data, list):
            return data[0] if (data and isinstance(data[0], dict)) else {}
        elif isinstance(data, dict):
            return data
        return {}
    except Exception:
        # Fallback regex jika ada teks pembuka/penutup tak terduga
        score_m = re.search(r'(?:suitability|suitable)\D{0,15}([0-9]*\.[0-9]+|[0-9]+)', response_str, re.IGNORECASE)
        summary_m = re.search(r'"summary":\s*"((?:[^"\\]|\\.)*)"', response_str)
        try:
            score_val = float(score_m.group(1)) if score_m else 0.0
        except ValueError:
            score_val = 0.0
        return {
            "summary": summary_m.group(1) if summary_m else "",
            "suitability": score_val
        }


# ==============================================================================
# PENENTUAN SKOR KESESUAIAN AD-HOC BUTTON (0 ATAU 1 DARI RESPON TEKS MODEL)
# ==============================================================================
def determine_button_suitability_score(response_text: str) -> float:
    """
    Menentukan skor suitability (0.0 atau 1.0) secara terprogram dari respon teks model
    untuk prompt ad-hoc button redirection (Seksi 5 Hlm. 13 artikel ilmiah) berdasarkan
    indikator positif/negatif yang paling pertama kali muncul (first occurrence).
    """
    text_lower = response_text.lower().strip()

    # Bersihkan pengulangan klausa prompt kondisional jika ada (misal: 'if/whether the text ... is right')
    cleaned_target = re.sub(r'\b(?:if|whether)\s+the\s+text\b[^\n.:]*?\b(?:is\s+)?right\b', '', text_lower)

    # Indikator negatif (ketidaksesuaian/kesalahan deskripsi tujuan tombol)
    negative_patterns = [
        r'\b(?:is\s+)?not\s+(?:right|correct|accurate|descriptive)\b',
        r'\bno\b',
        r'\bincorrect\b',
        r'\bplaceholder\b',
        r'\bpoor accessibility\b',
        r'\bdoes\s+not\s+(?:describe|indicate|tell)\b',
        r'\bnot\s+describe\b',
        r'\bwouldn\'t\s+know\b'
    ]

    # Indikator positif (kesesuaian deskripsi tujuan tombol)
    positive_patterns = [
        r'\byes\b',
        r'(?<!not\s)(?<!not\sbe\s)\b(?:the\s+text\s+)?(?:is\s+)?(?:correct|accurate)\b',
        r'(?<!not\s)(?<!not\sbe\s)\b(?:the\s+text\s+is|it\s+is)\s+right\b',
        r'(?<!not\s)\b(?:is\s+)?(?:the\s+)?right\s+(?:description|destination|text)\b',
        r'\bcorrectly\s+(?:indicates?|identifies?|describes?)\b',
        r'\baccurately\s+describes?\b'
    ]

    # Cari posisi kemunculan pertama masing-masing kategori
    first_neg_idx = min(
        [m.start() for pat in negative_patterns for m in [re.search(pat, cleaned_target)] if m],
        default=float('inf')
    )
    first_pos_idx = min(
        [m.start() for pat in positive_patterns for m in [re.search(pat, cleaned_target)] if m],
        default=float('inf')
    )

    # Keputusan berdasarkan indikator yang lebih dahulu muncul
    if first_pos_idx < first_neg_idx:
        return 1.0
    elif first_neg_idx < first_pos_idx:
        return 0.0

    return 0.0


# ==============================================================================
# PEMUATAN KONTEN HALAMAN SASARAN (BROWSE & FETCH)
# ==============================================================================
def clean_html_to_text(raw_html: str, max_chars: int = 3500) -> str:
    """Membersihkan kode HTML mentah menjadi teks esensial yang ringkas untuk LLM."""
    cleaned = re.sub(r'<(script|style|noscript)\b[^>]*>[\s\S]*?</\1>', ' ', raw_html, flags=re.I)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:max_chars]


def fetch_target_page_content(target: str):
    """
    Mengunjungi halaman/sumber daya sasaran (Browse & Fetch) dan mengambil kontennya.
    Mendukung berkas HTML lokal langsung serta URL web publik.
    """
    if not target:
        return "", "Target sasaran kosong"

    clean_target = str(target).strip().strip("'\"")

    # Jika target merupakan berkas lokal yang ada di filesystem
    if os.path.exists(clean_target):
        try:
            with open(clean_target, "r", encoding="utf-8") as f:
                content = f.read()
            return content, f"Berhasil membaca file lokal ({len(content)} karakter)"
        except Exception as e:
            return "", f"Gagal membaca file lokal {clean_target}: {e}"

    # Penanganan URL relatif W3C
    fetch_url = clean_target
    if clean_target.startswith("/"):
        fetch_url = f"https://www.w3.org{clean_target}"
    elif not clean_target.startswith("http://") and not clean_target.startswith("https://"):
        fetch_url = f"https://www.w3.org/WAI/{clean_target}"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "*/*"
    }

    try:
        resp = requests.get(fetch_url, headers=headers, timeout=12)
        if resp.status_code == 200:
            text_content = clean_html_to_text(resp.text)
            return text_content, f"Berhasil diakses via HTTP 200 ({len(text_content)} karakter teks)"
        else:
            fallback_text = f"Webpage at {clean_target}. Accessible destination resource."
            return fallback_text, f"HTTP {resp.status_code} saat mengakses {fetch_url}"
    except Exception as e:
        fallback_text = f"Webpage destination URL: {clean_target}."
        return fallback_text, f"Akses jaringan terkendala ({e})"


# ==============================================================================
# TOOL BROWSING WEB UNTUK LANGCHAIN AGENT (SESUAI PLAYWRIGHT DI GIST ARTIKEL)
# ==============================================================================
def create_browse_tool():
    """
    Membuat Tool LangChain untuk Agent agar dapat menjelajahi dan mengambil isi konten
    halaman URL atau berkas HTML lokal (menggantikan Playwright Browser Toolkit di Gist 23).
    """
    def _browse(target: str) -> str:
        clean_target = str(target).strip().strip("'\"")
        print(f"[Agent Tool: browse_webpage] Mengakses sasaran: {clean_target}")
        content, status = fetch_target_page_content(clean_target)
        print(f"[Agent Tool: browse_webpage] Status: {status}")
        return content

    return Tool(
        name="browse_webpage",
        func=_browse,
        description="Useful for browsing to a webpage URL or local HTML file path and retrieving its content."
    )


# ==============================================================================
# VARIABEL GLOBAL MODEL & AGENT (LAZY LOADING)
# ==============================================================================
EXTRACTOR_CHAIN = None
SUITABILITY_AGENT = None

# ==============================================================================
# INISIALISASI LLM DAN AUTONOMOUS AGENT (FOOTNOTE 23)
# ==============================================================================
def init_llm_and_agent():
    """Membangun dan memuat LLM Chain dan LangChain Agent sekali saja jika belum dimuat."""
    global EXTRACTOR_CHAIN, SUITABILITY_AGENT

    if EXTRACTOR_CHAIN is None or SUITABILITY_AGENT is None:
        model_name = "claude-sonnet-4-5-20250929"
        print("\n[INISIALISASI LLM ANTHROPIC CLAUDE & AGENT VIA LANGCHAIN]")
        print("-" * 80)
        print(f"Menghubungkan ChatAnthropic {model_name} (temperature=0)...")
        llm = ChatAnthropic(model=model_name, temperature=0)

        extractor_prompt = PromptTemplate(
            template=EXTRACTOR_TEMPLATE,
            input_variables=['html_code', 'special_instruction']
        )
        EXTRACTOR_CHAIN = LLMChain(prompt=extractor_prompt, llm=llm)

        # Inisialisasi LangChain Agent dengan Tool browse_webpage (persis seperti Gist Footnote 23):
        tools = [create_browse_tool()]
        SUITABILITY_AGENT = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True
        )

    return EXTRACTOR_CHAIN, SUITABILITY_AGENT


# ==============================================================================
# PIPELINE EKSEKUSI UTAMA EVALUASI TUNGGAL BERBASIS LLM
# ==============================================================================
def run_single_evaluation(file_path: str = HTML_FILE_PATH, use_adhoc: bool = USE_ADHOC):
    print("=" * 80)
    print("EVALUASI KASUS TUNGGAL WCAG SC 2.4.4 (EKSTRAKTOR BERBASIS PROMPT LLM)")
    print("=" * 80)
    print(f"Target File : {file_path}")
    print(f"Mode Ad-hoc : {'AKTIF (Modifikasi Prompt Ad-hoc)' if use_adhoc else 'NON-AKTIF'}")

    result = {
        "prompt_type": None,
        "ext_status": None,
        "link_url": None,
        "link_text": None,
        "page_summary": None,
        "suitability_score": None,
        "eval_status": None
    }

    # 1. Validasi Keberadaan Berkas HTML Kasus Uji
    if not os.path.exists(file_path):
        print(f"\n[ERROR] Berkas '{file_path}' tidak ditemukan!")
        result["eval_status"] = "ERROR"
        return result

    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # 2. Inisialisasi Rantai LLM & Agent LangChain (ChatAnthropic)
    extractor_chain, suitability_agent = init_llm_and_agent()

    # --------------------------------------------------------------------------
    # [TAHAP 1] Ekstraksi Elemen Tautan HTML Menggunakan LLM Prompt
    # --------------------------------------------------------------------------
    print("\nEKSTRAKSI ELEMEN TAUTAN BERBASIS PROMPT LLM")
    print("-" * 80)
    print("Kode HTML Kasus Uji:")
    for line in html_content.strip().splitlines():
        print(f"  | {line}")

    # Penentuan Petunjuk Khusus / Special Instruction (Seksi 5 Hlm. 12-13 Artikel Ilmiah)
    if not use_adhoc:
        prompt_type = "Base LLM Prompt"
        special_instruction = ""
    else:
        # Sisipan instruksi khusus ad-hoc prompt untuk elemen non-standar:
        if "aria-labelledby" in html_content or "aria-label" in html_content:
            prompt_type = "Ad-hoc Prompt (Link labelled by ARIA)"
            special_instruction = (
                "Notice that the link description is labelled by ARIA attributes (such as aria-label or aria-labelledby "
                "associating the link with a label element). Extract the accessible name/label as the link_text.\n"
            )
        elif "<area" in html_content or "usemap" in html_content:
            prompt_type = "Ad-hoc Prompt (Client-side image map <area>)"
            special_instruction = (
                "Notice that the link is defined using a client-side image map (<map> and <area> element). "
                "Extract the area href as link_url and alt text as link_text.\n"
            )
        elif "<button" in html_content or "role=\"link\"" in html_content or "window.location" in html_content:
            # Penanganan Kasus Khusus Button Redirection (Seksi 5 Hlm. 13 Artikel Ilmiah):
            # Pada artikel ilmiah aslinya, penulis menggunakan bantuan OpenInterpreter + GPT-4 secara terpisah
            # dengan prompt untuk mengecek arah navigasi tombol ("tell me if the text that describes where I will go if I click the button is right").
            # Di sini kita mereplikasi metodologi tersebut secara otomatis menggunakan suitability_agent dengan prompt ADHOC_BUTTON_PROMPT di Tahap 2.
            prompt_type = "Ad-hoc Prompt (Button redirection role='link')"
            special_instruction = (
                "Notice that the redirection to the link is triggered by a button element (e.g. role='link' or onclick redirection). "
                "Extract the redirection destination URL as link_url and the button text as link_text.\n"
            )
        else:
            prompt_type = "Standard Prompt"
            special_instruction = ""

    raw_extractor_output = extractor_chain.run({
        'html_code': html_content,
        'special_instruction': special_instruction
    })

    print("\nRespon Mentah LLM:")
    print(raw_extractor_output.strip())
    parsed_extractor = parse_extractor_json(raw_extractor_output)

    link_url = parsed_extractor.get("link_url")
    link_text = parsed_extractor.get("link_text")
    ext_status = "FOUND" if (link_url and link_text) else "NOT FOUND"

    result.update({
        "prompt_type": prompt_type,
        "ext_status": ext_status,
        "link_url": link_url,
        "link_text": link_text
    })

    print(f"\nHasil Parsing Ekstraksi:")
    print(f"  -> Tipe Prompt   : {prompt_type}")
    print(f"  -> Status        : {ext_status}")
    print(f"  -> Link URL      : {link_url if link_url else '-'}")
    print(f"  -> Link Text     : {link_text if link_text else '-'}")

    # Jika ekstraksi gagal pada Base Prompt karena menggunakan elemen non-standar
    if not link_url or not link_text:
        print("\nKEPUTUSAN AKHIR EVALUASI")
        print("-" * 80)
        print("Status Evaluasi : FAILED*")
        print("Penjelasan      : Elemen tautan/deskripsi tidak dapat diekstrak oleh Base Prompt karena")
        print("                  menggunakan pola non-standar (misalnya ARIA label, <area> image map,")
        print("                  atau tombol pengalihan <button role='link'>).")
        print("                  Aktifkan 'USE_ADHOC = True' atau gunakan opsi '--adhoc' untuk prompt modifikasi.")
        print("=" * 80)
        result["eval_status"] = "FAILED*"
        return result

    # --------------------------------------------------------------------------
    # [TAHAP 2] Peringkasan & Penilaian Kesesuaian via LangChain Agent (Gist Footnote 23)
    # --------------------------------------------------------------------------
    print("\nPERINGKASAN & PENILAIAN KESESUAIAN (LANGCHAIN AGENT DENGAN TOOL)")
    print("-" * 80)
    print(f"Target URL            : {link_url}")
    print(f"Deskripsi Tautan Uji  : \"{link_text}\"")

    # Penentuan Alur Evaluasi Tahap 2:
    # 1. Kasus Ad-hoc Button Redirection (Seksi 5 Hlm. 13 Artikel Ilmiah):
    #    Menggunakan suitability_agent dengan prompt halaman target langsung sesuai contoh artikel ilmiah,
    #    dan skor ditentukan sendiri (0 atau 1) dari respon model.
    # 2. Kasus Standar (Tautan/Anchor, ARIA, Image Map):
    #    Menggunakan suitability_agent dengan SUITABILITY_AGENT_PROMPT.
    if use_adhoc and prompt_type == "Ad-hoc Prompt (Button redirection role='link')":
        # Menggunakan berkas kasus uji lokal langsung
        target_page = file_path.replace("\\", "/")
        agent_input = ADHOC_BUTTON_PROMPT.format(target_page=target_page)

        print("\nMenjalankan Agent (Agent akan otomatis menggunakan tool 'browse_webpage' untuk membaca konten)...")
        raw_suitability_output = suitability_agent.run(agent_input)

        print("\nRespon Mentah Agent:")
        print(raw_suitability_output.strip())

        suitability_score = determine_button_suitability_score(raw_suitability_output)
        page_summary = raw_suitability_output.strip().replace("\n", " ")[:200]
    else:
        agent_input = SUITABILITY_AGENT_PROMPT.format(
            url=link_url,
            link_text=link_text
        )

        print("\nMenjalankan Agent (Agent akan otomatis menggunakan tool 'browse_webpage' untuk membaca konten)...")
        raw_suitability_output = suitability_agent.run(agent_input)

        print("\nRespon Mentah Agent:")
        print(raw_suitability_output.strip())

        parsed_suitability = parse_suitability_json(raw_suitability_output)
        page_summary = parsed_suitability.get("summary", "")
        suitability_score = float(parsed_suitability.get("suitability", 0.0))

    result["page_summary"] = page_summary
    result["suitability_score"] = suitability_score

    print(f"\nHasil Evaluasi:")
    print(f"  -> Ringkasan Halaman : \"{page_summary}\"")
    print(f"  -> Skor Kesesuaian   : {suitability_score:.2f}")

    # --------------------------------------------------------------------------
    # [TAHAP 3] Klasifikasi Keputusan Kepatuhan WCAG SC 2.4.4
    # --------------------------------------------------------------------------
    print("\nKEPUTUSAN AKHIR EVALUASI")
    print("-" * 80)
    is_passed = suitability_score >= THRESHOLD
    eval_status = "PASSED" if is_passed else "FAILED"
    result["eval_status"] = eval_status

    print(f"Status Evaluasi : {eval_status}")
    print(f"Ambang Batas    : Skor >= {THRESHOLD:.2f} untuk PASSED")
    print(f"Skor Diperoleh  : {suitability_score:.2f}")

    if is_passed:
        print("Kesimpulan      : Teks deskripsi tautan SESUAI secara semantik dengan konten")
        print("                  halaman tujuan (Memenuhi kriteria WCAG SC 2.4.4).")
    else:
        print("Kesimpulan      : Teks deskripsi tautan TIDAK SESUAI secara semantik dengan konten")
        print("                  halaman tujuan (Melanggar kriteria WCAG SC 2.4.4).")
    print("=" * 80)

    return result


if __name__ == "__main__":
    # Penggunaan via Terminal:
    # python evaluate_sc244_single.py testcases/sc244/selected/a8cc66de4d60e34c7ee0d09fd6ab965ac23d9b4f.html
    # python evaluate_sc244_single.py testcases/sc244/selected/a8cc66de4d60e34c7ee0d09fd6ab965ac23d9b4f.html --adhoc
    target_path = HTML_FILE_PATH
    adhoc_flag = USE_ADHOC

    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    if "--adhoc" in sys.argv:
        adhoc_flag = True

    run_single_evaluation(file_path=target_path, use_adhoc=adhoc_flag)
