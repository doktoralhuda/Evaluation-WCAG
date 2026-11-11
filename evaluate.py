"""
Skrip Batch Evaluasi WCAG SC 1.1.1, SC 2.4.4 & SC 3.1.2 (Modular)
Berdasarkan Metodologi Artikel Ilmiah:
"Turning manual web accessibility success criteria into automatic: an LLM-based approach"
(Pereira et al., Universal Access in the Information Society, 2024/2025)

Skrip ini mendukung evaluasi menggunakan Base Prompt:
1. WCAG SC 1.1.1 (Non-text Content)
2. WCAG SC 2.4.4 (Link Purpose in Context)
3. WCAG SC 3.1.2 (Language of Parts)
Dijalankan secara interaktif melalui antarmuka prompt CLI langkah demi langkah.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# DAFTAR ID KASUS UJI PER KRITERIA WCAG (TARGET_TESTCASE_IDS)
# ==============================================================================

TARGET_TESTCASE_IDS = {
    "1.1.1": [
        "8c29bcb24ac0f448846a2ffdad4c9693d5aef8c6",  # Example 1
        "b413c09531b239e27bcf79cb57302b429ef59fe6",  # Example 2
        "cab9b2d06e5a44e2056ccbdbb7096f55ab42859c",  # Example 3
        "7d97d6b2f3fa16760bf66026691281a8179f3260",  # Example 4
        "32bfac8a98cc212aa7bf9151bf40f665a7f51696",  # Example 5
        "38cc6a87fcc81fcc2248f0cd74ca48396b7aa432",  # Example 6 (CSS background-image)
        "40d83620b0bcbcf0e7380177384f48596823e7a9",  # Example 7
        "1b172036f8e219ef9b6f591d7f5df26e4ba11327",  # Example 8 (Tag <object>)
        "af4423575333947073fa3729f502ff0a0c6c2fbf"   # Example 9
    ],
    "2.4.4": [
        "a8cc66de4d60e34c7ee0d09fd6ab965ac23d9b4f",  # Example 1
        "ada7438401aba500eb03f678b05b9821a758336a",  # Example 2
        "d13a75a2a0b539a39063eb946505e3d3dd5aeef1",  # Example 3
        "4493c4b542c8e059e8423c77945ce5895428ab88",  # Example 4
        "d6a239059266b317de6a6e73dbf443c5ca8a6f5f",  # Example 5
        "5d16da98a4089b29ff76c611036c65e1c504c7bc",  # Example 6
        "e277de30edb9e550d8f9d5a72e1e3adde961d01d",  # Example 7
        "dee6c55162904cfb77c7f65614c4e6ae2baacea2",  # Example 8
        "b9a3949e2a7521698472a966c782434c4d9ce6fb",  # Example 9
        "d36abfa44924a4d4088bada05f439ae392dfd662",  # Example 10
        "b130285915a8ca42926a11553a5791f44b65d487",  # Example 11
        "a1e9ff296f0728e180aeb920beacb26bf88ddb12",  # Example 12
        "e4f70ef2843c6239d0bebe46b97a682bd901e749",  # Example 13
        "4e89fcc7903980482fe12350f864ca75963d6efd",  # Example 14
        "c6927fede2d5da439b2d346f39d2ec8980212b31",  # Example 15
        "e0d32d9583b2b545ca76295cff78e016a44854b6",  # Example 16
        "91abed1247fb6c9314457a6738343493056fe3bb",  # Example 17
        "8e6c190e0d2ba8f37707910bd1b984b6885ab548",  # Example 18
        "b55973d2f813b2fa7d0841202c13f65e41ca8823",  # Example 19
        "19d5c2888e4434b3e0fb2d9ea5818808e8380422"   # Example 20
    ],
    "3.1.2": [
        "53a78bad6e92791991df42c50d2e763a9f9d772c",  # Example 1
        "e0fe6824b5571e0552ab2955697c5ff0776abf79",  # Example 2
        "207782d0d8899521e2b51b5c384f83d7f4516358",  # Example 3
        "8376f95166a75a8541217b77ae6a235f1aac6c3d",  # Example 4
        "2febb4d398ed0d788f9ac054ff14cfbd68c0c1f1",  # Example 5
        "ec40c0a032b11cabc03d71b6884ab9b85ee160ad",  # Example 6
        "df9260fddb4d08ca0669bea363828d089b36317b",  # Example 7
        "5532e66ea71ed1f352f9911e224cbf290c7cc8e6",  # Example 8
        "53d05e6fdcc63ff61ef1e5ea8454eea318aa038a",  # Example 9
        "61c507e0aab456cce20538400fc1067be37953a0"   # Example 10
    ]
}

SC_CONFIG = {
    "1.1.1": {
        "title": "WCAG SC 1.1.1 (Non-text Content)",
        "score_label": "Sim",
        "dir_name": "sc111",
        "evaluator_module": "evaluate_sc111_single"
    },
    "2.4.4": {
        "title": "WCAG SC 2.4.4 (Link Purpose in Context)",
        "score_label": "Suit",
        "dir_name": "sc244",
        "evaluator_module": "evaluate_sc244_single"
    },
    "3.1.2": {
        "title": "WCAG SC 3.1.2 (Language of Parts)",
        "score_label": "Valid",
        "dir_name": "sc312",
        "evaluator_module": "evaluate_sc312_single"
    }
}


def run_evaluation_for_sc(sc_key: str, dataset_mode: str = "all", target_testcase_id: str = None):
    """
    Menjalankan batch evaluasi untuk kriteria keberhasilan tertentu (1.1.1, 2.4.4, atau 3.1.2).
    """
    config = SC_CONFIG.get(sc_key)
    if not config:
        print(f"[ERROR] Kriteria WCAG SC '{sc_key}' tidak dikenali.")
        return

    # Import modul evaluator secara dinamis sesuai kebutuhan
    if sc_key == "1.1.1":
        from evaluate_sc111_single import run_single_evaluation
    elif sc_key == "2.4.4":
        from evaluate_sc244_single import run_single_evaluation
    elif sc_key == "3.1.2":
        from evaluate_sc312_single import run_single_evaluation
    else:
        raise ValueError(f"Unsupported SC: {sc_key}")

    dir_name = config["dir_name"]
    score_label = config["score_label"]
    testcases_all = TARGET_TESTCASE_IDS.get(sc_key, [])

    # Filter kasus uji jika pengguna menentukan target_testcase_id
    if target_testcase_id:
        testcases = [tid for tid in testcases_all if target_testcase_id in tid]
        if not testcases:
            print(f"[ERROR] Kasus uji '{target_testcase_id}' tidak ditemukan dalam daftar SC {sc_key}.")
            return
    else:
        testcases = testcases_all

    if dataset_mode.lower() == "all":
        datasets = ["selected", "modified"]
    else:
        datasets = [dataset_mode.lower()]

    mode_label = "MODE BASE PROMPT"
    eval_mod_name = f"{config['evaluator_module']}.py"

    print("\n" + "=" * 92)
    print(f"BATCH EVALUASI OTOMATIS : {config['title']}")
    print(f"Evaluator Modul  : {eval_mod_name}")
    print(f"Konfigurasi      : {mode_label}")
    print(f"Dataset Target   : {', '.join([d.upper() for d in datasets])}")
    print(f"Jumlah Kasus Uji : {len(testcases)} kasus uji per-dataset")
    print("=" * 92)

    all_dataset_results = {}
    selected_results = {}

    for ds in datasets:
        is_selected = ds == "selected"
        expected_status = "PASSED" if is_selected else "FAILED"

        print("\n" + "=" * 92)
        print(f"MEMULAI DATASET: {ds.upper()} ({config['title']})")
        print(f"Direktori: testcases/{dir_name}/{ds}/")
        print("=" * 92)

        results = []

        for idx, test_id in enumerate(testcases, 1):
            file_path = os.path.join(BASE_DIR, "testcases", dir_name, ds, f"{test_id}.html").replace("\\", "/")

            print(f"\n>>> [{idx}/{len(testcases)}] Mengevaluasi {ds.upper()} - {test_id}.html ...")

            if not os.path.exists(file_path):
                print(f"[!] Berkas '{file_path}' tidak ditemukan, dilewati.")
                continue

            # Panggil fungsi evaluasi kasus tunggal secara modular (Base Prompt)
            eval_res = run_single_evaluation(file_path=file_path)
            eval_status = eval_res.get("eval_status")

            # Ambil skor kesesuaian (sim_score untuk SC 1.1.1, suitability_score untuk SC 2.4.4, score untuk SC 3.1.2)
            if "score" in eval_res and eval_res.get("score") is not None:
                score = eval_res.get("score")
            elif "sim_score" in eval_res:
                score = eval_res.get("sim_score")
            else:
                score = eval_res.get("suitability_score")

            # Penentuan Kolom A (Agreement) sesuai Seksi 4 & Tabel 3-6 artikel ilmiah
            if is_selected:
                agreement = "Y" if eval_status == expected_status else "-"
                selected_results[test_id] = eval_status
            else:
                sel_status = selected_results.get(test_id)
                if eval_status == "FAILED" and sel_status == "PASSED":
                    # True Negative: Model berhasil mendeteksi modifikasi/inkonsistensi
                    agreement = "Y"
                elif eval_status in ["FAILED", "FAILED*"] and sel_status == eval_status:
                    # Model menghasilkan kegagalan yang sama dengan kasus aslinya
                    agreement = "N"
                else:
                    agreement = "-"

            results.append({
                "no": idx,
                "id": test_id,
                "expected": expected_status,
                "score": score,
                "eval_status": eval_status,
                "agreement": agreement
            })

        all_dataset_results[ds] = results

    # ==========================================================================
    # TAMPILAN TABEL REKAPITULASI AKHIR
    # ==========================================================================
    print("\n" + "=" * 92)
    print(f"REKAPITULASI HASIL EVALUASI: {config['title']}")
    print(f"Konfigurasi Evaluasi: {mode_label}")
    print("=" * 92)

    for ds, results in all_dataset_results.items():
        is_selected = ds == "selected"
        table_ref = "KASUS ASLI W3C (SELECTED)" if is_selected else "KASUS MODIFIKASI (MODIFIED)"

        print("\n" + "-" * 92)
        print(f"{table_ref}")
        print("-" * 92)
        print(f"{'No':<3} | {'Testcase ID':<42} | {'Expected':<8} | {score_label:<6} | {'Result':<8} | {'Kolom A':<7}")
        print("-" * 92)

        for r in results:
            score_val = r["score"]
            score_str = f"{score_val:.2f}" if isinstance(score_val, (float, int)) else "-"
            print(f"{r['no']:<3} | {r['id']:<42} | {r['expected']:<8} | {score_str:<6} | {r['eval_status']:<8} | {r['agreement']:<7}")

        print("-" * 92)
        valid_count = sum(1 for r in results if r["eval_status"] == r["expected"])
        valid_pct = (valid_count / len(results)) * 100 if results else 0.0
        print(f"Valid Expected Percentage  : {valid_count}/{len(results)} * 100 = {valid_pct:.2f}% (Dibulatkan: {round(valid_pct)}%)")

        if not is_selected and selected_results:
            accurate_count = sum(1 for r in results if r["agreement"] == "Y")
            accurate_pct = (accurate_count / len(results)) * 100 if results else 0.0
            print(f"Accurate Expected Percentage: {accurate_count}/{len(results)} * 100 = {accurate_pct:.2f}% (Dibulatkan: {round(accurate_pct)}%)")
        print("-" * 92)


def safe_input(prompt_text: str, default: str = "") -> str:
    """Fungsi pembantu membaca input CLI dengan nilai default jika kosong."""
    try:
        val = input(prompt_text).strip()
        return val if val else default
    except (EOFError):
        return default


def main():
    print("\n" + "=" * 92)
    print("SISTEM BATCH EVALUASI AKSESIBILITAS WCAG DENGAN LLM & VLM")
    print("Berdasarkan Metodologi Penelitian Ilmiah Pereira et al. (2024/2025)")
    print("Petunjuk: Tekan [Ctrl+C] kapan saja untuk membatalkan dan keluar.")
    print("=" * 92)

    # --------------------------------------------------------------------------
    # LANGKAH 1: Pilih Mode Evaluasi (Berdasarkan SC atau Testcase ID)
    # --------------------------------------------------------------------------
    print("\nPilih Mode Evaluasi:")
    print("  [1] Berdasarkan Kriteria Keberhasilan (Success Criterion / SC)")
    print("  [2] Berdasarkan ID Kasus Uji Spesifik (Testcase ID)")
    print("-" * 92)
    mode_choice = safe_input("Masukkan pilihan mode [default: 1]: ", default="1")

    target_criteria = []
    target_testcase_id = None

    if mode_choice == "2":
        # Mode Berdasarkan Testcase ID
        print("\n" + "-" * 92)
        print("Pencarian ID Kasus Uji:")
        target_id_input = safe_input("Masukkan Testcase ID (contoh: 8c29bcb24ac0... atau ID parsial): ")
        if not target_id_input:
            print("[ERROR] Testcase ID tidak boleh kosong.")
            return

        target_testcase_id = target_id_input

        # Deteksi otomatis kriteria SC berdasarkan ID kasus uji
        matched_scs = []
        for sc, ids in TARGET_TESTCASE_IDS.items():
            if any(target_id_input in tid for tid in ids):
                matched_scs.append(sc)

        if len(matched_scs) == 1:
            target_criteria = matched_scs
            print(f"-> Kasus uji terdeteksi milik {SC_CONFIG[matched_scs[0]]['title']}.")
        elif len(matched_scs) > 1:
            print(f"-> Kasus uji ditemukan pada beberapa SC: {', '.join(matched_scs)}")
            print("Pilih SC yang ingin dievaluasi:")
            for idx, sc in enumerate(matched_scs, start=1):
                print(f"  [{idx}] {SC_CONFIG[sc]['title']}")
            sc_pick = safe_input("Pilih SC [default: 1]: ", default="1")
            try:
                target_criteria = [matched_scs[int(sc_pick) - 1]]
            except (ValueError, IndexError):
                target_criteria = [matched_scs[0]]
        else:
            print(f"[ERROR] Kasus uji '{target_id_input}' tidak ditemukan di daftar SC 1.1.1, SC 2.4.4, maupun SC 3.1.2.")
            return

    else:
        # Mode Berdasarkan Kriteria Keberhasilan (SC)
        print("\n" + "-" * 92)
        print("Pilih Kriteria Keberhasilan (Success Criterion) yang ingin dievaluasi:")
        sc_keys = list(SC_CONFIG.keys())
        for idx, key in enumerate(sc_keys, start=1):
            print(f"  [{idx}] {SC_CONFIG[key]['title']}")
        all_option_idx = len(sc_keys) + 1
        print(f"  [{all_option_idx}] All (Semua SC)")
        print("-" * 92)
        sc_choice = safe_input("Masukkan pilihan SC [default: 1]: ", default="1")

        try:
            choice_num = int(sc_choice)
            if 1 <= choice_num <= len(sc_keys):
                target_criteria = [sc_keys[choice_num - 1]]
            elif choice_num == all_option_idx:
                target_criteria = sc_keys.copy()
            else:
                target_criteria = [sc_keys[0]]
        except ValueError:
            target_criteria = [sc_keys[0]]

    # --------------------------------------------------------------------------
    # LANGKAH 2: Memilih Dataset
    # --------------------------------------------------------------------------
    print("\n" + "-" * 92)
    print("Pilih Dataset yang ingin dievaluasi:")
    print("  [1] Selected (Hanya kasus uji asli)")
    print("  [2] Modified (Hanya kasus uji modifikasi)")
    print("  [3] All (Selected & Modified)")
    print("-" * 92)
    ds_choice = safe_input("Masukkan pilihan dataset [default: 1]: ", default="1")

    if ds_choice == "2":
        dataset_mode = "modified"
    elif ds_choice == "3":
        dataset_mode = "all"
    else:
        dataset_mode = "selected"

    # --------------------------------------------------------------------------
    # EKSEKUSI BATCH EVALUASI (BASE PROMPT ONLY)
    # --------------------------------------------------------------------------
    for sc in target_criteria:
        run_evaluation_for_sc(
            sc_key=sc,
            dataset_mode=dataset_mode,
            target_testcase_id=target_testcase_id
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Proses evaluasi dihentikan oleh pengguna (Ctrl+C). Keluar...")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

