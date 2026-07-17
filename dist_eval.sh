export OMP_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=2
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=4 custom_eval.py --config config.new_eval
