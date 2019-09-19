import SimpleITK as sitk
import env
import os

mha_path = os.path.join(env.TRAIN_IMAGES_DIR, '50655.mha')
mha_img = sitk.ReadImage(mha_path)
mha_arr = sitk.GetArrayFromImage(mha_img)

level = 50  # 窗位
window = 350  # 窗宽
windowMinimum = level - window / 2 # 设定窗位窗宽后，这就是对应的最小强度值和最大强度值
windowMaximum = level + window / 2
mha_arr[mha_arr < windowMinimum] = windowMinimum
mha_arr[mha_arr > windowMaximum] = windowMaximum

# 保存图片（mha文件无法保存窗位窗宽信息，保存了用itk-snap打开还是原来的样子）
mha_img = sitk.GetImageFromArray(mha_arr)
save_path = os.path.join(env.TRAIN_PROCESSED_IMAGES_DIR, '50655.mha')
sitk.WriteImage(mha_img, save_path)