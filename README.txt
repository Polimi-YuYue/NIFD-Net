1. 预训练单模态分支 (Pre-training)
你需要分别训练 Vibration (SVS) 和 Acoustic (NCS) 分支。
# Pretrain Vibration (SVS)
python pretrain.py --modality SVS --data_root /path/to/Vibration_Dataset

# Pretrain Acoustic (NCS)
python pretrain.py --modality NCS --data_root /path/to/Acoustic_Dataset

2. 训练 NIFD-Net (Training)
在两个预训练模型生成后，运行主程序。

python train.py --svs_data /path/to/Vibration_Dataset \
                --ncs_data /path/to/Acoustic_Dataset \
                --batch_size 16 \
                --epochs_stage2 100