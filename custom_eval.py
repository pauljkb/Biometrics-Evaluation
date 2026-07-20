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

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Fixed input sizes at eval time -> let cuDNN pick the fastest conv algorithms
# instead of the (slower) default deterministic ones.
if device == "cuda":
    torch.backends.cudnn.benchmark = True


def load_model(cfg):
    # Load checkpoint straight onto the target device -- avoids the extra
    # CPU round-trip you get from torch.load(...) followed by .to(device).
    weights = torch.load(cfg.checkpoint, map_location=device)

    net = get_model(cfg.network, **cfg)
    net.load_state_dict(weights)
    net = net.to(device)
    net.eval()

    # DataParallel only helps when you actually have >1 GPU to split batches
    # across -- on a single GPU (or CPU) it's pure overhead: every forward
    # call replicates the model and scatters/gathers through Python. Only
    # wrap when it can actually parallelize something.
    n_gpus = torch.cuda.device_count()
    if device == "cuda" and n_gpus > 1:
        model = torch.nn.DataParallel(net)
    else:
        model = net

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
    with torch.inference_mode():
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
            ijb_eval(0, model, target="IJBB", eval_path=cfg.ijbb_path, config=cfg)

        if "IJBC" in cfg.eval_type:
            logging.info("--- IJBC Evaluation ---")
            ijb_eval(0, model, target="IJBC", eval_path=cfg.ijbc_path, config=cfg)

        if "FLOPS" in cfg.eval_type:
            input_size = (3, cfg.image_size, cfg.image_size)
            inputs = torch.randn(1, 3, 112, 112).cuda()

            print("---------------  PTFLOPS  ---------------")
            from ptflops import get_model_complexity_info
            macs, params = get_model_complexity_info(
                                model, input_size, as_strings=False,
                                print_per_layer_stat=False, verbose=False
                            )
            gmacs = macs / (1000**3)
            print("%.3f GFLOPs"%gmacs)
            print("%.3f Mparams"%(params/(1000**2)))


            #print("---------------  FVCORE  ---------------")
            #from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
            #flops = FlopCountAnalysis(model, inputs)
            #params = parameter_count(model)
            #print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
            #print(flop_count_table(flops, max_depth=4))
            #print(f"Params: {params[''] / 1e6:.2f} M")


            print("---------------  DEEPSPEED  ---------------")
            from deepspeed.profiling.flops_profiler import get_model_profile
            flops, macs, params = get_model_profile(
                model=model,
                args=(inputs,),
                print_profile=True,
                detailed=False,
                module_depth=-1,
            )

            print("---------------  calflops  ---------------")
            from calflops import calculate_flops
            flops, macs, params = calculate_flops(model=model, 
                                                input_shape=(1,3,112,112),
                                                output_as_string=True,
                                                output_precision=4)
            print("FLOPs:%s   MACs:%s   Params:%s \n" %(flops, macs, params))


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