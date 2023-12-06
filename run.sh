# python3 dataset/split.py
# python3 dataset/split.py --balance true
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29000 run.py --run_mode train
# CUDA_VISIBLE_DEVICES=0 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29000 run.py --run_mode eval
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29002 run.py --run_mode train --balance true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29002 run.py --run_mode eval --balance true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29002 run.py --run_mode train --balance true --quantize true --cpu true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29002 run.py --run_mode eval --balance true --quantize true --cpu true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode train --balance true --quantize true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode eval --balance true --quantize true
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode train --balance true --quantize true --cpu true --load_full_ckpt true
CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode eval --balance true --quantize true --cpu true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode train --balance true --quantize true --lora true
# CUDA_VISIBLE_DEVICES=1 torchrun --nnodes 1 --nproc_per_node 1 --master_port 29003 run.py --run_mode eval --balance true --quantize true --lora true
# python3 test_single.py