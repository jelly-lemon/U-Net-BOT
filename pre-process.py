import SimpleITK as sitk
import os
import env

# 读入原始CT图像
images_names = os.listdir(env.TRAIN_IMAGES_DIR)
for file_name in images_names:
    # 读取图片
    img_path = os.path.join(env.TRAIN_IMAGES_DIR, file_name)
    #print(img_path)
    mha_img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(mha_img)   # ndarray类型的
    #print(type(img_arr))
    #print(img_arr.shape)

    # 调整窗位窗宽
    level = 50      # 窗位
    window = 256    # 窗宽
    # 窗位窗宽的另一种说法
    windowMinimum = level - window / 2
    windowMaximum = level + window / 2
    img_arr[img_arr < windowMinimum] = windowMinimum
    img_arr[img_arr > windowMaximum] = windowMaximum
    # 保存原shape，待会儿用来恢复原状
    original_shape = img_arr.shape
    img_arr = img_arr.ravel()
    num_pixel = img_arr.shape[0]

    # 修改像素值
    for i in range(num_pixel):
        img_arr[i] = (256 / (windowMaximum - windowMinimum)) * (img_arr[i] - windowMinimum)

    img_arr.reshape(original_shape)
    mha_img = sitk.GetImageFromArray(img_arr)


    #sitk.IntensityWindowing(mha_img, windowMinimum=level - window / 2, windowMaximum=level + window / 2)

    # 保存图片（mha文件无法保存窗位窗宽信息，保存了用itk-snap打开还是原来的样子）
    sitk.WriteImage(mha_img, os.path.join(env.TRAIN_PROCESSED_IMAGES_DIR, file_name))

    break

# 读入人工标注的CT图像
masks_names = os.listdir(env.TRAIN_MASKS_DIR)
# for file_name in masks_names:
#     # 读取图片
#     img_path = os.path.join(env.TRAIN_MASKS_DIR, file_name)
#     print(img_path)
#     mha_img = sitk.ReadImage(img_path)
#
#     # TODO 滤波平滑处理
#
#
#     # 保存图片
#     sitk.WriteImage(mha_img, os.path.join(r'D:\2-ITK-SNAP\data\train\processed_masks', file_name))
#
#     break




