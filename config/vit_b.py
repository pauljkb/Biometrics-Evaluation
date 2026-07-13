from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "vit_b"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128
config.lr = 0.02
config.verbose = 2000
config.dali = False


# EVAL
config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 93431
config.num_image = 5179510
config.num_epoch = 20
config.warmup_epoch = 0
config.eval_type = ["FR"]
config.val_targets_fr = ['lfw', 'cfp_fp', "agedb_30"]
config.eval_path = "./bins"
config.output = "./output"
config.log_eval = False
config.normalize_type = "01"
config.interpolation_type = "bilinear"
config.image_size = 112
config.model_name = "vit_finetune"
config.model_path = "./checkpoints/vitb_8reg_ms1mv3.pt"
config.num_register_token = 8
config.batch_size_eval = 128
config.model_folder = "./checkpoints"