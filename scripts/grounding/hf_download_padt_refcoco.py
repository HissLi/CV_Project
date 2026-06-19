from huggingface_hub import hf_hub_download

FILES = ["refcoco_val.json", "refcoco+_val.json", "refcocog_val.json"]

for filename in FILES:
    path = hf_hub_download(
        repo_id="PaDT-MLLM/RefCOCO",
        repo_type="dataset",
        filename=filename,
        local_dir="/home/turing_lab/cse12210210/cv_project/datasets/refer/padt",
        local_dir_use_symlinks=False,
    )
    print("DOWN", path)
