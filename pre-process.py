import SimpleITK as sitk
import os
import env
# TODO 读入人工标注的CT图像
masks_names = os.listdir(env.TRAIN_MASKS_DIR)

for file_name in masks_names:
    # 读取图片
    img_path = os.path.join(env.TRAIN_MASKS_DIR, file_name)
    mha_img = sitk.ReadImage(img_path)

    # 调整窗位窗宽
    window = 400
    level = 50
    sitk.IntensityWindowing(mha_img, windowMinimum=level-window/2, windowMaximum=level+window/2)

    # 滤波平滑处理
    sitk.WriteImage(mha_img, os.path.join(r'C:\Users\laptop lemon', file_name))

    break

# TODO 调整窗位窗宽

# TODO 双边滤波处理

# TODO 保存处理后的CT图像到指定文件夹，保存格式为mha


