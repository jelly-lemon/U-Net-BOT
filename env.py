import os
import platform

"""
数据的存放目录应该是这个样子的
--data
  |--train  # 训练用的图像
  |  |--images  # 原始图像
  |  |--masks   # 标注图像
  |--predict    # 预测用的图像
  |  |--images  # 需要预测的图像
  |  |--results # 预测结果
  |--models # 训练好的模型
"""

# 数据的根目录
DATA_DIR = (r'D:\2-ITK-SNAP\data' if platform.system() == "Windows" else "/home/data_new/zhangyongqing/model/data")

# 用来训练的原始CT图像文件夹路径
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'images')

# 用来训练的人工标注CT图像文件夹路径
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, 'train', 'masks')

# 模型保存路径
MODEL_SAVE_DIR = os.path.join(DATA_DIR, 'models')

# TODO 处理后的原始CT图像文件夹路径
TRAIN_PROCESSED_IMAGES_DIR = os.path.join(DATA_DIR, 'train', 'processed_images')

# TODO 处理后（调整窗位窗宽、平滑滤波）的人工标注CT图像文件夹路径
TRAIN_PROCESSED_MASKS_DIR = os.path.join(DATA_DIR, 'train', 'processed_masks')

# TODO 待预测的CT图像文件夹路径
PREDICT_IMAGES_DIR = ''

# TODO 预测结果输出路径
PREDICT_OUTPUT_DIR = ''

