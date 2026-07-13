from easydict import EasyDict as edict
# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()

# Margin Base Softmax
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.save_all_states = False
config.output = "ms1mv3_arcface_r50"

config.embedding_size = 512

# Partial FC
config.sample_rate = 1
config.interclass_filtering_threshold = 0

config.fp16 = False
config.batch_size = 128

# For SGD 
config.optimizer = "sgd"
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4

# For AdamW
# config.optimizer = "adamw"
# config.lr = 0.001
# config.weight_decay = 0.1

config.verbose = 2000
config.frequent = 10

# For Large Sacle Dataset, such as WebFace42M
config.dali = False 
config.dali_aug = False

# Gradient ACC
config.gradient_acc = 1

# setup seed
config.seed = 2048

# dataload numworkers
config.num_workers = 2

# WandB Logger
config.wandb_key = "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
config.suffix_run_name = None
config.using_wandb = False
config.wandb_entity = "entity"
config.wandb_project = "project"
config.wandb_log_all = True
config.save_artifacts = False
config.wandb_resume = False
config.rand_augment = True # Keypoint paper augmentation
 # resume wandb run: Only if the you wand t resume the last run that it was interrupted
# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

### SPECIFIC
config.use_adaface_loss = False
config.adaface_m = 0.4
config.adaface_h = 0.333
config.adaface_t_alpha = 0.01
config.margin_list = (1.0, 0.0, 0.4)
config.network = "vit_s"
config.resume = False
config.output = "/home/chettaou/workspace/output"
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.weight_decay = 0.05
config.batch_size = 64
config.scheduler_type = "else" #!
config.optimizer = "adamw"
config.lr = 0.001
config.img_size = 112
config.verbose = 1000
config.dali = False
config.num_register_token = 8

config.dataset = "CASIA"
config.rec = "/home/chettaou/workspace/data/" + config.dataset
if config.dataset == "MS1MV2":
    config.num_classes = 85742
    config.num_image = 5822653
elif config.dataset == "MS1MV3":
    config.num_classes = 93431
    config.num_image = 5179510
elif config.dataset == "MS1MV3_sub_10k":
    config.num_classes = 10000
    config.num_image = 584012
elif config.dataset == "MS1MV3_sub_20k":
    config.num_classes = 20000
    config.num_image = 1131045
elif config.dataset == "WEBFACE4M":
    config.rec = "/home/chettaou/workspace/data/" + "webface4m_112x112.lmdb_dataset"
    config.num_classes = 205990
    config.num_image = 4235242
elif config.dataset == "CASIA":
    config.rec = "/home/chettaou/workspace/data/casia_training"
    config.num_classes = 10000
    config.num_image = 500000




config.num_epoch = 3
config.warmup_epoch = config.num_epoch // 10
config.val_rec = "/home/chettaou/workspace/data/validation"
config.val_targets = ["lfw"] # , "cfp_fp", "agedb_30"] # ["lfw", "cfp_fp", "agedb_30"] # "lfw", "cfp_fp", "cfp_ff", "agedb_30", "calfw", "cplfw" / ["lfw", "cfp_fp", "agedb_30"] 
