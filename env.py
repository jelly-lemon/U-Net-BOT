import os
"""有没有一个方法，代码自动判断是哪一台电脑，然后根据电脑加载不同的值？"""
#COMPUTER = 'lm'

# TODO 数据的根目录
DATA_DIR = r'D:\2-ITK-SNAP\data'

# TODO 用来训练的原始CT图像文件夹路径
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')

# 处理后的原始CT图像文件夹路径
TRAIN_PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'processed_images')

# TODO 用来训练的人工标注CT图像文件夹路径
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train', 'masks')

# TODO 处理后（调整窗位窗宽、平滑滤波）的人工标注CT图像文件夹路径
TRAIN_PROCESSED_MASKS_DIR = os.path.join(DATA_DIR, 'train', 'processed_masks')

# TODO 待预测的CT图像文件夹路径
PREDICT_IMAGES_DIR = ''

# TODO 预测结果输出路径
PREDICT_OUTPUT_DIR = ''

# TODO 模型保存路径
MODEL_SAVE_DIR = ''