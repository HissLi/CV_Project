# Model Weights (Local Only)

Large pretrained weights are not stored in git. Download them before running grounding eval:

```bash
# Grounding DINO base
huggingface-cli download IDEA-Research/grounding-dino-base --local-dir models/gdino

# OWL-ViT base-patch32
huggingface-cli download google/owlvit-base-patch32 --local-dir models/owlvit
```

Or use `scripts/grounding/setup_local_refcoco.sh` on a machine with network access.
