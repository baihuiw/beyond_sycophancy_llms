import json

def clean_uid(path, suffix):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            js = json.loads(line)
            uid = js["custom_id"]
            # 保留 v1, v2, v3...
            base = uid.split("-")[0]
            js["custom_id"] = base
            out.append(js)
    out_path = path.replace(".jsonl", f"_clean.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print("cleaned →", out_path)
    return out_path


if __name__ == "__main__":
    clean_uid("final_experiments/results_global/output_global_countrylist_Global.jsonl", "countrylist")
    clean_uid("final_experiments/results_global/output_global_counts_Global.jsonl", "counts")
    clean_uid("final_experiments/results_global/output_global_simple_Global.jsonl", "simple")
