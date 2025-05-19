from pathlib import Path

from DFINE.tools.inference.trt_inf import TRTInference
from Server.profile_inference import profile_inference

# ── paths & constants ─────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TENSOR_MODEL = PROJECT_ROOT / "DFINE" / "model" / "model.engine"
DEVICE = "cuda:0"
ENGINE = TRTInference(TENSOR_MODEL)
profile_inference(ENGINE, "2k.jpg", device="cuda:0")
