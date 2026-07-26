"""
Virtual Try-On (Qwen-Image-Edit, no-mask outfit transfer) — Replicate cog predictor.

Upload a PERSON photo + a GARMENT photo. The person is shown wearing that outfit,
their real face/pose/background preserved, and the garment's SHAPE changes
(e.g. a shirt becomes a full kurta). Full-image edit, no clothing mask.

Built on replicate/cog-comfyui. Workflow: workflow_qwen.json
"""

import os
import json
import shutil
import subprocess
from cog import BasePredictor, Input, Path

from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP = "/tmp/comfyui_temp"
WORKFLOW_PATH = "workflow_qwen.json"

# Models downloaded at setup time (build-time HF pget hits a xet-CDN 403).
# (hf repo, filename in repo, local dest under ComfyUI/models)
HF_MODELS = [
    ("Comfy-Org/Qwen-Image-Edit_ComfyUI",
     "split_files/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors",
     "ComfyUI/models/diffusion_models/qwen_image_edit_2509_fp8_e4m3fn.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
     "ComfyUI/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI",
     "split_files/vae/qwen_image_vae.safetensors",
     "ComfyUI/models/vae/qwen_image_vae.safetensors"),
    ("lightx2v/Qwen-Image-Lightning",
     "Qwen-Image-Lightning-8steps-V1.0.safetensors",
     "ComfyUI/models/loras/Qwen-Image-Lightning-8steps.safetensors"),
]
# The clothes try-on LoRA is on Civitai (direct URL works with pget).
CIVITAI_LORA = ("https://civitai.com/api/download/models/2196278",
                "ComfyUI/models/loras/qwen_clothes_tryon.safetensors")


def _download_models():
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    for repo, fname, dest in HF_MODELS:
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        p = hf_hub_download(repo_id=repo, filename=fname)
        shutil.copy(p, dest)
    url, dest = CIVITAI_LORA
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.check_call(["pget", url, dest])


class Predictor(BasePredictor):
    def setup(self):
        _download_models()
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def predict(
        self,
        person_image: Path = Input(
            description="A photo of the person to dress. Their real face, pose and background are preserved.",
        ),
        garment_image: Path = Input(
            description="A photo of the outfit/garment to put on the person (works best on a plain background).",
        ),
        seed: int = Input(
            description="Random seed. Leave at 0 for a random result.",
            default=0,
        ),
    ) -> Path:
        self.comfyUI.cleanup(ALL_DIRECTORIES=[OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP])
        if seed == 0:
            seed = int.from_bytes(os.urandom(3), "big")

        person_name = "person_input.png"
        garment_name = "garment_input.png"
        shutil.copy(str(person_image), os.path.join(INPUT_DIR, person_name))
        shutil.copy(str(garment_image), os.path.join(INPUT_DIR, garment_name))

        with open(WORKFLOW_PATH, "r") as f:
            wf = json.load(f)
        wf.pop("_comment", None)

        # node 10 = garment LoadImage, node 11 = person LoadImage, node 50 = KSampler
        wf["10"]["inputs"]["image"] = garment_name
        wf["11"]["inputs"]["image"] = person_name
        wf["50"]["inputs"]["seed"] = seed

        wf = self.comfyUI.load_workflow(wf)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        files = self.comfyUI.get_files(OUTPUT_DIR)
        if not files:
            raise RuntimeError("No output produced.")
        return files[-1]
