from huggingface_hub import list_repo_files

REPOS = [
    "PaDT-MLLM/RefCOCO",
    "jxu124/refcoco",
    "yiqun/referit",
]

for repo in REPOS:
    try:
        files = list_repo_files(repo, repo_type="dataset")
        print(f"REPO {repo} FILES {len(files)}")
        for path in files[:80]:
            print(f"  {path}")
    except Exception as exc:
        print(f"REPO {repo} ERR {type(exc).__name__}: {exc}")
