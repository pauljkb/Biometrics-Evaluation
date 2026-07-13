import logging
import os
import sys
import torch
import numpy as np
import warnings
import onnxruntime as ort

sys.path.append(os.path.join(os.getcwd()))

warnings.filterwarnings("ignore", message="xFormers is not available*")

from config.vit_b import config as cfg
from backbone import get_model
from utils.transform import transform_image
from utils.evaluation import CallBackVerification
from utils.utils_logging import init_logging

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model():
    # Load Model
    model = get_model(0, **cfg)

    if cfg.model_path is not None:
        if cfg.model_name == "resnet":
            print("Loading model from path: " + cfg.model_path)
            model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
            model = model.to(device)

        elif cfg.model_name == "vit_finetune":
            print("Loading model from path: " + cfg.model_path)
            checkpoint = torch.load(cfg.model_path, map_location="cpu")
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model = model.to(device)

        elif cfg.model_name == "onnx":
            class ONNXModelWrapper(torch.nn.Module):
                def __init__(self, onnx_path: str, device: str = "cpu"):
                    super().__init__()
                    self.device = device
                    print(f"Running on device: {self.device}")

                    providers = (
                        ["CUDAExecutionProvider"]
                        if device.startswith("cuda")
                        else ["CPUExecutionProvider"]
                    )

                    self.session = ort.InferenceSession(onnx_path, providers=providers)
                    self.input_name = self.session.get_inputs()[0].name
                    self.output_name = self.session.get_outputs()[0].name

                def forward(self, x: torch.Tensor):
                    x_np = x.detach().cpu().numpy().astype(np.float32)
                    output = self.session.run([self.output_name], {self.input_name: x_np})[0]
                    return torch.from_numpy(output).to(self.device)

            print("Loading ONNX model from path: " + cfg.model_path)
            model = ONNXModelWrapper(cfg.model_path, device=device)
            model.eval()

        else:
            print("Loading model from path: " + cfg.model_path)
            model.backbone.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
            model = model.backbone.to(device)
    else:
        model = model.backbone.to(device)

    model.eval()

    return model


def evaluate():
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)

    if cfg.log_eval:
        log_root = logging.getLogger()
        init_logging(log_root, 0, cfg.output, logfile=cfg.log_name)

        for key, value in cfg.items():
            num_space = 25 - len(key)
            logging.info(": " + key + " " * num_space + str(value))

    # Transform
    transform = transform_image(
        image_size=cfg.image_size,
        normalize_type=cfg.normalize_type,
        interpolation_type=cfg.interpolation_type
    )

    # Load model
    model = load_model()

    # Eval
    if "FR" in cfg.eval_type:
        logging.info("--- Small Benchmarks Evaluation ---")
        callback_verification = CallBackVerification(
            5, 0, cfg.val_targets_fr, cfg.eval_path,
            cfg.image_size, transform, cfg.batch_size_eval, cfg.model_name)

        result = callback_verification(4, model)
        print(result)


if __name__ == "__main__":
    evaluate()