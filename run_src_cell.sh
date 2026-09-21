#!/bin/bash
#CUDA_VISIBLE_DEVICES=0,1 /media/zhouzhihao/DeepInformatic_dataset/lihansheng/envs/torch110/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 bjrt_train.py --output_dir=bjrt_cell_20250220_resnet50_eos_coef_0_5.pth  --eos_coef=0.5 --dataset=dataset --num_classes=4 --num_workers=4 --start_eval=50 --epochs=400 --batch_size=4 --lr=4e-5
#CUDA_VISIBLE_DEVICES=0,1 /home/songlinru/anaconda3/envs/torch110/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 bjrt_train.py --output_dir=bjrt_cell_20250220_resnet50_eos_coef_0_5.pth  --eos_coef=0.5 --dataset=dataset --num_classes=4 --num_workers=4 --start_eval=50 --epochs=400 --batch_size=4 --lr=4e-5
#CUDA_VISIBLE_DEVICES=0,1 /media/zhouzhihao/DeepInformatic_dataset/lihansheng/envs/torch110/bin/python train.py --output_dir=taimo_cell_202450122_resnet50_eos_coef_0_5.pth --eos_coef=0.5 --dataset=dataset --num_classes=1 --num_workers=4 --start_eval=10 --epochs=600 --batch_size=8 --lr=2e-5

# tests
#CUDA_VISIBLE_DEVICES=0,1 /media/zhouzhihao/DeepInformatic_dataset/lihansheng/envs/torch110/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 train.py  --eos_coef=0.8 --dataset=dataset --num_classes=4 --num_workers=4 --start_eval=50 --epochs=400 --batch_size=4
# CUDA_VISIBLE_DEVICES=2,6 python train_p2p.py --output_dir /home/data/p2p-src-zzh-2025/pth/冰冻_2025  --eos_coef=0.5 --dataset=冰冻-2025 --num_classes=1 --num_workers=0 --start_eval=30 --epochs=40 --batch_size=2 --lr=4e-5

# -----------------训练
CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 train_p2p.py \
    --output_dir /home/data/p2p-src-zzh-2025/pth/冰冻_2025_1_0.3_full \
    --eos_coef=0.5 \
    --dataset=冰冻-2025 \
    --num_classes=1 \
    --num_workers=0 \
    --start_eval=30 \
    --epochs=200 \
    --batch_size=2 \
    --lr=4e-5 \
    --resume /home/data/p2p-src-zzh-2025/pth/冰冻_2025_1_0.3_full/recent_model.pth

# # ----------------测试
# CUDA_VISIBLE_DEVICES=2 python train_p2p.py \
#     --output_dir /home/data/ \
#     --eos_coef=0.6 \
#     --dataset=冰冻-2025 \
#     --num_classes=1 \
#     --num_workers=0 \
#     --start_eval=30 \
#     --epochs=100 \
#     --batch_size=2 \
#     --lr=4e-5