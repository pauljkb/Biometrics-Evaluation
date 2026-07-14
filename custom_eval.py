import logging
import os
import sys
import torch
import numpy as np
import warnings
import cv2
import onnxruntime as ort
from tqdm import tqdm
import argparse
sys.path.append(os.path.join(os.getcwd()))
import importlib
warnings.filterwarnings("ignore", message="xFormers is not available*")

from backbone import get_model
from utils.transform import transform_image
from utils.evaluation import CallBackVerification
from utils.utils_logging import init_eval_logging
from utils.validate_ijb import ijb_eval
from utils.validate_tinyface import tinyface_eval

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_model(cfg):
    # Load Model

    weights = torch.load(cfg.checkpoint)

    net = get_model(cfg.network, **cfg)
    net.load_state_dict(weights)
    model = torch.nn.DataParallel(net)
    model = model.to(device)
    model.eval()

    return model


def evaluate(cfg):
    if not os.path.exists(cfg.output):
        os.makedirs(cfg.output)

    if cfg.log_eval:
        log_root = logging.getLogger()
        init_eval_logging(0, cfg.output, logfile=cfg.get("log_name", "eval.log"))

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
    model = load_model(cfg)

    # Eval
    if "FR" in cfg.eval_type:
        logging.info("--- Small Benchmarks Evaluation ---")
        callback_verification = CallBackVerification(
            5, 0, cfg.val_targets_fr, cfg.bin_path,
            cfg.image_size, transform, cfg.batch_size_eval, cfg.network)

        callback_verification(4, model)

    if "FR-Bias" in cfg.eval_type:
        logging.info("--- Small Benchmarks Bias Evaluation ---")
        callback_verification = CallBackVerification(
            5, 0, cfg.val_targets_bias, cfg.bin_path,
            cfg.image_size, transform, cfg.batch_size_eval, cfg.network)

        callback_verification(4, model)

    if "TinyFace" in cfg.eval_type:
        logging.info("--- TinyFace Evaluation ---")
        tinyface_eval(0, model, **cfg)

    if "IJBB" in cfg.eval_type:
        logging.info("--- IJBB Evaluation ---")
        ijb_eval(0, model, target="IJBB", eval_path=cfg.ijbb_path, **cfg)

    if "IJBC" in cfg.eval_type:
        logging.info("--- IJBC Evaluation ---")
        ijb_eval(0, model, target="IJBC", eval_path=cfg.ijbc_path, **cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval Model")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config for eval')
    args = parser.parse_args()

    cfg_module = importlib.import_module(args.config)
    cfg = cfg_module.config

    evaluate(cfg)