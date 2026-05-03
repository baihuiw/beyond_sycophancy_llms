import os
import shutil
import re

def safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[OK] Created directory: {path}")
    else:
        print(f"[SKIP] Directory already exists: {path}")

def safe_move(src, dst):
    if os.path.exists(src):
        print(f"[MOVE] {src}  →  {dst}")
        shutil.move(src, dst)
    else:
        print(f"[MISS] {src} not found, skipping.")

def extract_country_name(filename):
    """
    From e.g. 'distribution_Austria.jsonl' → 'Austria'
              'mean_alone_New_Zealand.jsonl' → 'New_Zealand'
    """
    match = re.search(r".*?_(.+)\.jsonl$", filename)
    if match:
        return match.group(1)
    return None

def main():
    print("\n=== Organizing repository ===\n")

    FE = "final_experiments"
    DATA = "data"

    # 1) Create high-level folders
    safe_mkdir(f"{FE}/results_global")
    safe_mkdir(f"{FE}/results_by_country")
    safe_mkdir(f"{FE}/archive/batch_files")
    safe_mkdir(f"{FE}/archive/logs")
    safe_mkdir(f"{DATA}/ground_truth")
    safe_mkdir("deprecated")
    safe_mkdir("utils")

    # 2) Move global files → results_global/
    global_outputs = [
        "output_global_countrylist_Global_clean.jsonl",
        "output_global_counts_Global_clean.jsonl",
        "output_global_simple_Global_clean.jsonl",
        "global_merged_clean.jsonl",
    ]
    for f in global_outputs:
        safe_move(f"{FE}/{f}", f"{FE}/results_global/{f}")

    # 3) Move batch files → archive/batch_files/
    for fname in os.listdir(FE):
        if fname.endswith("_batch_file.jsonl"):
            safe_move(f"{FE}/{fname}", f"{FE}/archive/batch_files/{fname}")

    # 4) Move logs → archive/logs/
    for fname in ["experiment_log.txt", "run_distributional_alignment.log"]:
        safe_move(f"{FE}/{fname}", f"{FE}/archive/logs/{fname}")

    # 5) Move ground truth files
    gt_files = [
        "ISSP Distributional Alignment Dataset (1).xlsx",
        "ISSP Distributional Alignment Dataset Non-Discrete Scales.xlsx",
        "ZA10000_v2-0-0.sav",
        "question_participants_by_country.json",
    ]
    for f in gt_files:
        safe_move(f"{DATA}/{f}", f"{DATA}/ground_truth/{f}")

    # 6) Move deprecated scripts → deprecated/
    deprecated_scripts = [
        "run_distributional_alignment.py",
        "run_all_conditions.py",
        "run_all_countries.py",
        "run_all_jsonl.py",
        "run_experiment.sh",
    ]
    for f in deprecated_scripts:
        safe_move(f, f"deprecated/{f}")

    # 7) NEW: Move all single-country files → results_by_country/{country}/
    print("\n=== Organizing per-country files ===\n")

    for fname in os.listdir(FE):
        if not fname.endswith(".jsonl"):
            continue

        # Skip global files (already moved)
        if "Global" in fname:
            continue

        country = extract_country_name(fname)
        if not country:
            continue

        country_dir = f"{FE}/results_by_country/{country}"
        safe_mkdir(country_dir)

        safe_move(f"{FE}/{fname}", f"{country_dir}/{fname}")

    print("\n=== Done! Repository now organized. ===\n")

if __name__ == "__main__":
    main()

