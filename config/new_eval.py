from easydict import EasyDict as edict


config = edict()

# Model settings
config.fp16 = False
config.num_features = 512
config.num_register_token = 8
config.image_size = 112
config.batch_size_eval = 128
config.normalize_type = "arcface" # "01", "-1_1", "arcface", "imagenet", "clip"
config.interpolation_type = "bicubic" # "nearest", "bilinear", "bicubic", "area", "lanczos"

# NEW
config.checkpoint = "checkpoints/vitb_8reg_ms1mv3.pt"
config.network = "vit_b"
config.output = "./logs"
config.log_eval = True
config.log_name = "run.log"
config.eval_desc = "eval_desc"

# EVAL
config.bin_path = "./bins"
config.eval_type = ["FR"] # FR, FR-Bias, TinyFace, IJBB, IJBC
config.val_targets_fr = ['lfw']# "calfw", "cplfw"]
config.val_targets_bias = ["african_test", "asian_test", "caucasian_test", "indian_test"]

# IJBC/IJBB
config.ijbb_path = "/igd/a1/Share/Chettaoui/Evaluation/IJBB"
config.ijbc_path = "/igd/a1/Share/Chettaoui/Evaluation/IJBC"
config.use_detector_score = True
config.use_norm_score = True
config.use_flip_test = True

#TinyFace
config.tinyface_path = "/igd/a1/Share/Chettaoui/Evaluation"
config.fusion_method = "norm_weighted_avg"

