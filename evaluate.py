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

from config.config_eval import config as cfg
from backbone import get_model
from data.transform import transform_image
from utils.evaluation import CallBackVerification, CallBackVerificationBias
from utils.logging import init_logging
from finetuning import apply_lora_model
from utils.evaluate_cifar import evaluate_model
from utils.validate_tinyface import tinyface_eval
from utils.validate_mad import mad_eval
from utils.validate_pad import pad_eval
from utils.evaluate_bfw import evaluate_model_bfw_csv
# from utils.validate_ijbs import ijbs_eval
# from utils.validate_scface import scface_eval
from utils.validate_ijb import ijb_eval


def load_model(local_rank):
    # Load Model
    model = get_model(local_rank, **cfg)

    # Attach LoRA layers
    if cfg.use_lora:
        apply_lora_model(local_rank, model, **cfg)

    # Load Trained model
    if cfg.model_path is not None:
        if cfg.model_name == "resnet":
            print("Loading model from path: " + cfg.model_path)
            model.load_state_dict(torch.load(cfg.model_path))
            model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank],
                                            find_unused_parameters=False)
        elif cfg.model_name == "vit_finetune":
            print("Loading model from path: " + cfg.model_path)
            checkpoint = torch.load(cfg.model_path)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
            new_state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
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
            model = ONNXModelWrapper(cfg.model_path, device="cuda")
            model.eval()
        else:
            print("Loading model from path: " + cfg.model_path)
            model.backbone.load_state_dict(torch.load(cfg.model_path))
            model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank],
                                            find_unused_parameters=False)
    else:
        model = DistributedDataParallel(module=model.backbone, broadcast_buffers=False, device_ids=[local_rank],
                                        find_unused_parameters=False)

    model.eval()

    return model


def evaluate(args):
    dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(seconds=7200000))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)

    if not os.path.exists(cfg.output) and local_rank == 0:
        os.makedirs(cfg.output)

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

        if cfg.model_folder is not None:
            def extract_number(path):
                match = re.search(r'/(\d+)backbone\.pth$', path)
                return int(match.group(1)) if match else -1

            model_paths = glob.glob(os.path.join(cfg.model_folder, "*.pth"))
            model_paths = [path for path in model_paths if "header" not in os.path.basename(
                path).lower()]  # Filter out files that contain "header" in the name
            model_paths = sorted(model_paths, key=extract_number)

            for model_path in model_paths:
                cfg.model_path = model_path
                model = load_model(local_rank)
                if cfg.model_name == "baseline_insight" and cfg.use_index_token is not None:
                    results = []
                    for index in range(model.module.num_patches):
                        model.module.use_index_token = index
                        result = callback_verification(4, model)
                        result = round(np.mean(list(result.values())) * 100, 2)
                        results.append(result)
                    print(len(results))
                    print(results)
                else:
                    callback_verification(4, model)

        if cfg.model_name == "baseline_insight" and cfg.use_index_token is not None:
            results = []

            for index in range(model.module.num_patches - model.module.add_register_token):
                model.module.use_index_token = index
                result = callback_verification(4, model)
                result = round(np.mean(list(result.values())) * 100, 2)
                results.append(result)
            print(len(results))
            print(results)
        else:
            callback_verification(4, model)

    if "BiasText" in cfg.eval_type:
        logging.info("--- BiasText Evaluation ---")
        callback_verification = CallBackVerificationBias(
            5, local_rank, cfg.val_targets_bt, cfg.eval_path,
            cfg.image_size, transform, cfg.batch_size_eval, cfg.model_name, cfg.fusion_type_bt)
        callback_verification(4, model)

    if "BFW" in cfg.eval_type:
        logging.info("--- BFW Evaluation ---")
        results = evaluate_model_bfw_csv(
            model=model,
            transform=transform,
            csv_path=cfg.csv_path,
            img_root=cfg.img_root,
            fusion_type=cfg.fusion_type_bfw,
            model_name=cfg.model_name,
        )

    if "TinyFace" in cfg.eval_type:
        logging.info("--- TinyFace Evaluation ---")
        if cfg.model_name == "clip":
            model = model.module.visual

        tinyface_eval(local_rank, model, **cfg)

    if "IJBB" in cfg.eval_type:
        logging.info("--- IJBB Evaluation ---")
        if cfg.model_name == "clip":
            model = model.module.visual
        ijb_eval(local_rank, model, target="IJBB", **cfg)

    if "IJBC" in cfg.eval_type:
        logging.info("--- IJBC Evaluation ---")
        if cfg.model_name == "clip":
            model = model.module.visual
        ijb_eval(local_rank, model, target="IJBC", **cfg)

    if "SCface" in cfg.eval_type:
        pass  # scface_eval(local_rank, model, **cfg)

    if "IJB-S" in cfg.eval_type:
        pass  # ijbs_eval(local_rank, model, **cfg)

    if "Generate_embeddings" in cfg.eval_type:
        if not os.path.exists(cfg.output_embeddings_path) and local_rank == 0:
            os.makedirs(cfg.output_embeddings_path)

        if cfg.model_name == "clip":
            model = model.module.visual

        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        with torch.no_grad():
            image_paths = []
            for root, dirs, files in os.walk(cfg.images_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_extensions:
                        img_path = os.path.join(root, file)
                        if os.path.isfile(img_path):
                            image_paths.append(img_path)

            for img_path in tqdm(image_paths, desc="Processing images"):
                # read image
                img = cv2.imread(img_path)
                if img is None:
                    print(img_path)
                img = transform(img).unsqueeze(0).to("cuda")

                # image embedding
                embedding = model(img).detach().cpu().squeeze().numpy()

                # save embedding
                subdir = os.path.basename(os.path.dirname(img_path))  # "005840"
                filename = os.path.splitext(os.path.basename(img_path))[0]  # "00318924"
                subdir_path = os.path.join(cfg.output_embeddings_path, subdir)
                output_path = os.path.join(cfg.output_embeddings_path, subdir, filename)
                if not os.path.exists(subdir_path) and local_rank == 0:
                    os.makedirs(subdir_path)
                np.save(output_path, embedding)

    if "General" in cfg.eval_type:
        evaluate_model(
            model=model, transform=transform, test_data_list=cfg.test_datasets, batch_size=cfg.batch_size_eval,
            normalization_type=cfg.cifar_normalization_type, use_open_clip_model=cfg.use_open_clip_model,
            use_custom_transform=cfg.use_custom_transform, open_clip_variant=cfg.open_clip_variant,
            linear_probe_evaluation=cfg.linear_probe_evaluation, data_path=cfg.eval_path
        )

    if "MAD" in cfg.eval_type:
        mad_eval(
            model=model, test_data=cfg.test_datasets, test_data_path=cfg.test_data_path, transform=transform,
            normalization_type=cfg.normalization_type, use_open_clip_model=cfg.use_open_clip_model,
            header_path=cfg.header_path,
            batch_size=cfg.batch_size_eval
        )

    if "PAD" in cfg.eval_type:
        pad_eval(
            model=model, test_data=cfg.test_datasets, test_data_path=cfg.test_data_path, transform=transform,
            normalization_type=cfg.normalization_type, use_open_clip_model=cfg.use_open_clip_model,
            header_path=cfg.header_path,
            batch_size=cfg.batch_size_eval
        )

    if "FLOPS" in cfg.eval_type:
        input_size = (3, cfg.image_size, cfg.image_size)
        inputs = torch.randn(1, 3, 112, 112).cuda()

        print("---------------  PTFLOPS  ---------------")
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        gmacs = macs / (1000 ** 3)
        print("%.3f GFLOPs" % gmacs)
        print("%.3f Mparams" % (params / (1000 ** 2)))

        # print("---------------  FVCORE  ---------------")
        # from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
        # flops = FlopCountAnalysis(model, inputs)
        # params = parameter_count(model)
        # print(f"FLOPs: {flops.total() / 1e9:.2f} GFLOPs")
        # print(flop_count_table(flops, max_depth=4))
        # print(f"Params: {params[''] / 1e6:.2f} M")

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
                                              input_shape=(1, 3, 112, 112),
                                              output_as_string=True,
                                              output_precision=4)
        print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluation job')
    parser.add_argument('--local-rank', type=int, help='local_rank')
    parser.add_argument('--debug', default=False, type=bool, help='Log additional debug informations')
    args = parser.parse_args()

    evaluate(args)
