import logging
import os
import sys
import torch
import numpy as np
import warnings
import cv2
import onnxruntime as ort
from tqdm import tqdm

sys.path.append(os.path.join(os.getcwd()))

warnings.filterwarnings("ignore", message="xFormers is not available*")

from config.vit_b import config as cfg
from backbone import get_model
from utils.transform import transform_image
from utils.evaluation import CallBackVerification
from utils.utils_logging import init_logging
from utils.validate_ijb import ijb_eval

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

    if "BiasText" in cfg.eval_type:
        logging.info("--- BiasText Evaluation ---")
        callback_verification = CallBackVerificationBias(
            5, 0, cfg.val_targets_bt, cfg.eval_path,
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
            model = model.module.visual if hasattr(model, "module") else model

        tinyface_eval(0, model, **cfg)

    if "IJBB" in cfg.eval_type:
        logging.info("--- IJBB Evaluation ---")
        if cfg.model_name == "clip":
            model = model.module.visual if hasattr(model, "module") else model
        ijb_eval(0, model, target="IJBB", **cfg)

    if "IJBC" in cfg.eval_type:
        logging.info("--- IJBC Evaluation ---")
        if cfg.model_name == "clip":
            model = model.module.visual if hasattr(model, "module") else model
        ijb_eval(0, model, target="IJBC", **cfg)

    if "SCface" in cfg.eval_type:
        pass  # scface_eval(0, model, **cfg)

    if "IJB-S" in cfg.eval_type:
        pass  # ijbs_eval(0, model, **cfg)

    if "Generate_embeddings" in cfg.eval_type:
        if not os.path.exists(cfg.output_embeddings_path):
            os.makedirs(cfg.output_embeddings_path)

        if cfg.model_name == "clip":
            model = model.module.visual if hasattr(model, "module") else model

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
                    continue
                img = transform(img).unsqueeze(0).to(device)

                # image embedding
                embedding = model(img).detach().cpu().squeeze().numpy()

                # save embedding
                subdir = os.path.basename(os.path.dirname(img_path))  # "005840"
                filename = os.path.splitext(os.path.basename(img_path))[0]  # "00318924"
                subdir_path = os.path.join(cfg.output_embeddings_path, subdir)
                output_path = os.path.join(cfg.output_embeddings_path, subdir, filename)
                if not os.path.exists(subdir_path):
                    os.makedirs(subdir_path, exist_ok=True)
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
        inputs = torch.randn(1, 3, 112, 112).to(device)

        print("---------------  PTFLOPS  ---------------")
        from ptflops import get_model_complexity_info
        macs, params = get_model_complexity_info(
            model, input_size, as_strings=False,
            print_per_layer_stat=False, verbose=False
        )
        gmacs = macs / (1000 ** 3)
        print("%.3f GFLOPs" % gmacs)
        print("%.3f Mparams" % (params / (1000 ** 2)))

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
    evaluate()