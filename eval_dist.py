import logging
import os
import glob
import re
import sys
import torch
import argparse
import time
import numpy as np
import cv2
import warnings
import torch.distributed as dist
import onnxruntime as ort

from datetime import timedelta
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join(os.getcwd()))

warnings.filterwarnings("ignore", message="xFormers is not available*")

from config.vit_b import config as cfg
from backbone import get_model
from utils.transform import transform_image
from utils.evaluation import CallBackVerification
from utils.utils_logging import init_logging


def load_model(local_rank):
    # Load Model
    model = get_model(local_rank, **cfg)

    if cfg.model_path is not None:
        if cfg.model_name == "resnet":
            print("Loading model from path: " + cfg.model_path)
            model.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
            model = model.to(local_rank)
            model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank],
                                            find_unused_parameters=False)

        elif cfg.model_name == "vit_finetune":
            print("Loading model from path: " + cfg.model_path)
            checkpoint = torch.load(cfg.model_path, map_location="cpu")
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model = model.to(local_rank)
            model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank],
                                            find_unused_parameters=False)

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
            # ONNXRuntime uses its own CUDA device selection, not torch DDP.
            model = ONNXModelWrapper(cfg.model_path, device=f"cuda:{local_rank}")
            model.eval()

        else:
            print("Loading model from path: " + cfg.model_path)
            model.backbone.load_state_dict(torch.load(cfg.model_path, map_location="cpu"))
            model = model.backbone.to(local_rank)
            model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank],
                                            find_unused_parameters=False)
    else:
        model = model.backbone.to(local_rank)
        model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank],
                                        find_unused_parameters=False)

    model.eval()

    return model


def evaluate(local_rank):
    if not os.path.exists(cfg.output) and local_rank == 0:
        os.makedirs(cfg.output)

    # Make sure rank 0 has created the dir before others might rely on it (e.g. logging)
    dist.barrier()

    if cfg.log_eval:
        log_root = logging.getLogger()
        init_logging(log_root, local_rank, cfg.output, logfile=cfg.log_name)

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
    model = load_model(local_rank)

    # Eval
    if "FR" in cfg.eval_type:
        logging.info("--- Small Benchmarks Evaluation ---")
        callback_verification = CallBackVerification(
            5, local_rank, cfg.val_targets_fr, cfg.eval_path,
            cfg.image_size, transform, cfg.batch_size_eval, cfg.model_name)

        result = callback_verification(4, model)
        if local_rank == 0:
            print(result)

    # (other eval_type branches unchanged, omitted here for brevity — keep as in your original)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation job')
    parser.add_argument('--local-rank', type=int, help='local_rank')
    parser.add_argument('--debug', default=False, type=bool, help='Log additional debug informations')
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])

    # Bind this process to its assigned GPU BEFORE creating the process group
    # or moving any tensors — this is what was missing.
    torch.cuda.set_device(local_rank)

    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    dist.init_process_group(backend)

    evaluate(local_rank)

    dist.destroy_process_group()