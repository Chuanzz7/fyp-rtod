from pathlib import Path

import onnxruntime as ort

so = ort.SessionOptions()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONNX_MODEL = PROJECT_ROOT / "DFINE" / "model" / "dfine_x_obj365.onnx"
sess = ort.InferenceSession(str(ONNX_MODEL),
                            sess_options=so,
                            providers=[
                                ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),  # Enable FP16
                                "CUDAExecutionProvider",
                                "CPUExecutionProvider"
                            ])
