# Deploy the Virtual Try-On model to Replicate

This model is built on top of **replicate/cog-comfyui**. The files in this
folder are the *custom overlay* — they must be dropped into a fork of that repo,
because the build needs cog-comfyui's own code (`comfyui.py`,
`weights_downloader.py`, the ComfyUI submodule, etc.).

## Files in this overlay
| File | Purpose |
|---|---|
| `predict.py` | The app: face upload + garment upload + outfit/background dropdowns → result. Replaces cog-comfyui's predict.py. |
| `cog.yaml` | Build config (torch, PuLID/CatVTON deps). Merge into their cog.yaml. |
| `custom_nodes.json` | The 5 node packs to install (PuLID, KJNodes, Flux-TryOff, LayerStyle ×2). Replaces theirs. |
| `weights.json` | Model weights (Flux, PuLID, realism LoRA, EVA-CLIP). Register in their weights system. |
| `workflow_combined.json` | The ComfyUI node graph (PuLID → optional CatVTON). Ship alongside predict.py. |
| `scripts/patch_pulid.sh` | Applies the required `**kwargs` patch to the PuLID node (see PATCHES.md). Call in the build. |

## Deploy steps (Replicate builds on THEIR servers — nothing heavy on your Mac)

1. **Fork cog-comfyui** on GitHub: https://github.com/replicate/cog-comfyui → Fork
   (into `gokul3sedb/cog-comfyui` or a new repo).

2. **Overlay these files** into the fork's root:
   - Replace `predict.py`, `custom_nodes.json`.
   - Add `workflow_combined.json`, `scripts/patch_pulid.sh`.
   - Merge `cog.yaml` python_packages into the fork's cog.yaml.
   - Add the `patch_pulid.sh` call + the `facenet-pytorch` then `torch==2.6.0`
     reinstall as `run:` steps in cog.yaml (see PATCHES.md).
   - Register the `weights.json` entries in cog-comfyui's supported-weights list
     (see their MAKING_A_MODEL_GUIDE.md) OR let them runtime-download.

3. **Create the model on Replicate**: replicate.com → Create model →
   name it (e.g. `gokul3sedb/virtual-tryon`) → hardware **Nvidia A100 (80GB)** or
   **H100** (needs the VRAM for Flux+PuLID+CatVTON).

4. **Connect the GitHub repo** to the model (Replicate → model → Settings →
   connect GitHub) → trigger a build. Replicate builds the image on their
   infrastructure.

5. **First build is slow** (downloads ~60GB of models). Watch the build logs;
   fix any custom-node dep errors (the usual: a node's requirements.txt).

6. **Once built**, the model page IS the app: face upload, garment upload,
   outfit + background dropdowns, Run.

## Known issues to expect in the build (from our H100 testing)
- **PuLID**: needs `facenet-pytorch`; that pip-downgrades torch → reinstall
  `torch==2.6.0` after. And apply the `**kwargs` patch (PATCHES.md).
- **CatVTON model** (`xiaozaa/catvton-flux-alpha`, ~24GB) auto-downloads at first
  run — ensure enough disk in the Replicate image / a big enough volume.
- **LayerStyle** pulls heavy deps (segmentation) — build can be slow.

## Calling it via API (after it's live)
Use the Replicate API token (the `r8_...` one) with the `replicate` Python client:
```python
import replicate
out = replicate.run("gokul3sedb/virtual-tryon:VERSION",
    input={"face_image": open("face.jpg","rb"),
           "outfit_preset": "formal", "background_preset": "office"})
```
