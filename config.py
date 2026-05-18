# config.py
import torch

# 数据路径
DATA_ROOT = r"D:\CAT_IN_HQU"

# 超参数
BATCH_SIZE = 32
NUM_EPOCHS = 40
LR = 1e-4
IMG_SIZE = 224

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 保存路径
MODEL_SAVE_PATH = "cat_classifier.pth"
HISTORY_SAVE_PATH = "history.json"