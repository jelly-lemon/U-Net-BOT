"""
对图像进行预处理
调整病人CT图像的窗位窗宽
对人工标注进行双边平滑滤波
"""
import SimpleITK as sitk
import os
import env


def adjust_img_level_window():
    """
    调整CT图像的窗位窗宽，把像素强度低于minimum的都改为minimum，把大于maximum都改为maximum
    读取指定路径的mha文件，调整后，另存到指定路径中
    """
    # 读入原始CT图像
    images_names = os.listdir(env.TRAIN_IMAGES_DIR)
    for file_name in images_names:
        # 读取图片
        img_path = os.path.join(env.TRAIN_IMAGES_DIR, file_name)
        mha_img = sitk.ReadImage(img_path)
        img_arr = sitk.GetArrayFromImage(mha_img)  # ndarray类型，每个数据的type是什么呢？

        # 调整窗位窗宽（pixel intensity < minimum的变成纯黑，pixel intensity > maximux的变成纯白）
        # 相当于去噪？
        level = 50  # 窗位
        window = 350  # 窗宽
        window_minimum = level - window / 2  # 设定窗位窗宽后，这就是对应的最小强度值和最大强度值
        window_maximum = level + window / 2
        img_arr[img_arr < window_minimum] = window_minimum
        img_arr[img_arr > window_maximum] = window_maximum

        # 保存图片
        mha_img = sitk.GetImageFromArray(img_arr)
        save_name = os.path.splitext(file_name)[0] + ' - processed.mha'
        sitk.WriteImage(mha_img, os.path.join(env.TRAIN_PROCESSED_IMAGES_DIR, save_name))

# TODO 对人工标注图像进行滤波处理
def filter_masks():
    # 读入人工标注的CT图像
    masks_names = os.listdir(env.TRAIN_MASKS_DIR)
    for file_name in masks_names:
        # 读取图片
        img_path = os.path.join(env.TRAIN_MASKS_DIR, file_name)
        print(img_path)
        mha_img = sitk.ReadImage(img_path)

        # TODO 滤波平滑处理


        # 保存图片
        sitk.WriteImage(mha_img, os.path.join(r'D:\2-ITK-SNAP\data\train\processed_masks', file_name))



adjust_img_level_window()



