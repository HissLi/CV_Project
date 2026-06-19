import json

BASE = "/home/turing_lab/cse12210210/cv_project/datasets/refer/padt"
FILES = ["refcoco_val.json", "refcoco+_val.json", "refcocog_val.json"]

for filename in FILES:
    path = f"{BASE}/{filename}"
    with open(path, "r", encoding="utf-8") as f:
        lines = [f.readline().strip() for _ in range(3)]
    print(filename, "first_nonempty_lines", sum(1 for x in lines if x))
    obj = json.loads(lines[0])
    print("keys", list(obj.keys()))
    print("sample", {k: obj[k] for k in list(obj.keys())[:8]})
