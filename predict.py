"""
Virtual Try-On — Replicate cog predictor.

Two modes (auto-selected by whether a garment is uploaded):
  A) Preset outfit: face_image + outfit_preset + background_preset
       -> Flux + PuLID (identity) + realism LoRA generate the dressed person.
  B) Garment upload: face_image + garment_image (+ background_preset)
       -> generate a base person, then CatVTON swaps the uploaded garment on.

Built on replicate/cog-comfyui. See workflow_combined.json for the node graph.
"""

import os
import json
import shutil
from typing import Optional
from cog import BasePredictor, Input, Path

from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP = "/tmp/comfyui_temp"
WORKFLOW_PATH = "workflow_combined.json"

# ---- Preset maps: dropdown choice -> prompt fragment -----------------------
OUTFIT_PRESETS = {
    "casual": "a casual white cotton t-shirt and blue denim jacket with jeans",
    "formal": "a tailored charcoal grey three-piece business suit, white dress shirt and silk tie",
    "ethnic": "a traditional embroidered cream silk kurta with gold thread work",
    "seasonal": "a chunky knit wool sweater and a padded parka jacket with fur-lined hood",
    "streetwear": "an oversized graphic hoodie and cargo pants",
    "smart casual": "a fitted navy blazer over a plain white t-shirt with chinos",
}
BACKGROUND_PRESETS = {
    "studio (default)": "against a clean neutral studio backdrop, soft even lighting",
    "city street": "on a busy city street at golden hour",
    "office": "in a modern minimalist office lobby",
    "outdoor nature": "outdoors in a green park with soft natural daylight",
    "cafe": "sitting in a cozy cafe with warm ambient light",
    "beach": "on a sunny beach with the ocean behind, bright daylight",
    "festive venue": "at a decorated festive venue with warm celebratory lighting",
}

REALISM_TEMPLATE = (
    "candid amateur iPhone photo of {subject} wearing {outfit}, {background}, "
    "natural available light, natural skin texture with visible pores and subtle "
    "imperfections, slight film grain, realistic, unposed snapshot, upper body"
)


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

    def predict(
        self,
        face_image: Path = Input(
            description="A clear, front-facing photo of the person's face. Their identity is preserved in the result.",
        ),
        outfit_preset: str = Input(
            description="Pick an outfit style. Ignored if you upload a garment image.",
            choices=list(OUTFIT_PRESETS.keys()),
            default="casual",
        ),
        background_preset: str = Input(
            description="Pick a background/location.",
            choices=list(BACKGROUND_PRESETS.keys()),
            default="studio (default)",
        ),
        garment_image: Optional[Path] = Input(
            description="Optional: upload a photo of a specific garment (e.g. a shirt on a plain background) to try that exact item on. Overrides the outfit preset.",
            default=None,
        ),
        seed: int = Input(
            description="Random seed. Leave at 0 for a random result.",
            default=0,
        ),
    ) -> Path:
        self.comfyUI.cleanup(ALL_DIRECTORIES=[OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP])
        if seed == 0:
            seed = int.from_bytes(os.urandom(3), "big")

        # --- stage input images into ComfyUI's input dir ---
        face_name = "face_input.png"
        shutil.copy(str(face_image), os.path.join(INPUT_DIR, face_name))
        use_garment = garment_image is not None
        garment_name = None
        if use_garment:
            garment_name = "garment_input.png"
            shutil.copy(str(garment_image), os.path.join(INPUT_DIR, garment_name))

        # --- build the prompt from presets ---
        # For garment-upload mode we still generate a plausible base outfit so the
        # body/pose exist; CatVTON then replaces the upper garment.
        base_outfit = OUTFIT_PRESETS["casual"] if use_garment else OUTFIT_PRESETS[outfit_preset]
        background = BACKGROUND_PRESETS[background_preset]
        positive = REALISM_TEMPLATE.format(subject="a person", outfit=base_outfit, background=background)

        # --- load + patch the workflow ---
        with open(WORKFLOW_PATH, "r") as f:
            wf = json.load(f)
        wf.pop("_comment", None)
        for k in list(wf.keys()):
            if k.startswith("_"):
                wf.pop(k)

        wf["13"]["inputs"]["image"] = face_name          # PuLID face reference
        wf["3"]["inputs"]["text"] = positive             # positive prompt
        wf["6"]["inputs"]["seed"] = seed                 # sampler seed

        if use_garment:
            # Mode B: keep the CatVTON branch; drop the person-only SaveImage (8).
            wf["20"]["inputs"]["image"] = garment_name
            wf["25"]["inputs"]["seed"] = seed
            wf.pop("8", None)
        else:
            # Mode A: preset outfit only. Drop the entire CatVTON branch (20-26).
            for n in ["20", "21", "22", "23", "24", "25", "26"]:
                wf.pop(n, None)

        # --- run ---
        wf = self.comfyUI.load_workflow(wf)
        self.comfyUI.connect()
        self.comfyUI.run_workflow(wf)

        # --- return the single output image ---
        files = self.comfyUI.get_files(OUTPUT_DIR)
        if not files:
            raise RuntimeError("No output produced.")
        return files[-1]
