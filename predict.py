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
from cog import BasePredictor, Input, Path

from comfyui import ComfyUI

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP = "/tmp/comfyui_temp"
WORKFLOW_PATH = "workflow_qwen.json"


class Predictor(BasePredictor):
    def setup(self):
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
