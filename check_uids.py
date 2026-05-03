import json

def load_uid(path):
    uids = []
    with open(path) as f:
        for line in f:
            js = json.loads(line)
            uids.append(js["custom_id"])
    return set(uids)

u1 = load_uid("final_experiments/results_global/output_global_countrylist_Global.jsonl")
u2 = load_uid("final_experiments/results_global/output_global_counts_Global.jsonl")
u3 = load_uid("final_experiments/results_global/output_global_simple_Global.jsonl")

print("countrylist:", len(u1))
print("counts:", len(u2))
print("simple:", len(u3))

print("intersection:", len(u1 & u2 & u3))
print("missing_in_counts:", u1 - u2)
print("missing_in_simple:", u1 - u3)

