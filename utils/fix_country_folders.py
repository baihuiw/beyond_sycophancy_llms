import os
from pathlib import Path
import shutil
import re

BASE = Path("final_experiments/results_by_country")

def extract_country(filename):
    """
    Extract country name from formats like:
    - distribution_Australia.jsonl
    - mean_alone_Australia.jsonl
    - std_extended_United_States.jsonl
    """
    # Remove .jsonl
    name = filename.replace(".jsonl", "")
    parts = name.split("_")

    # Find the last part that starts a country name
    # Example: ["std", "extended", "United", "States"]
    # We want: "United_States"
    for i in range(len(parts)):
        if parts[i][0].isupper():
            return "_".join(parts[i:])

    return None


def main():
    print("\n=== Organizing files by country ===\n")

    if not BASE.exists():
        print("❌ BASE folder not found:", BASE)
        return

    files = [p for p in BASE.rglob("*.jsonl") if p.is_file()]

    if not files:
        print("❌ No JSONL files found!")
        return

    print(f"Found {len(files)} JSONL files.\n")

    for f in files:
        country = extract_country(f.name)

        if not country:
            print(f"⚠ Could NOT extract country from: {f.name}")
            continue

        # Final folder name (cleaned)
        country_clean = country.replace("_", " ")
        target_folder = BASE / country_clean

        if not target_folder.exists():
            target_folder.mkdir()

        target_path = target_folder / f.name

        print(f"→ Moving {f.name}  →  {target_folder}/")

        shutil.move(str(f), str(target_path))

    print("\n=== Cleaning up leftover empty folders ===\n")
    for folder in BASE.iterdir():
        if folder.is_dir() and not any(folder.iterdir()):
            print("Removing empty folder:", folder)
            folder.rmdir()

    print("\n🎉 DONE! Country folders reorganized successfully!\n")

if __name__ == "__main__":
    main()




