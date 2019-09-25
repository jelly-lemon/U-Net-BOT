"""
利用训练好的unet模型，对mha图像进行预测
"""
import time
from model import Unet
from data_reader import mha_data_reader
import SimpleITK as sitk
import env
import os
import gzip


def gzip_compress_and_save(original_file: str, gzip_file: str):
    """
    读取一个文件，压缩后另存为.gz文件
    :param original_file:原文件路径
    :param gzip_file:压缩文件路径
    :return:无
    """
    with open(original_file, 'rb') as original_file:
        with gzip.open(gzip_file, 'wb') as zip_file:
            zip_file.writelines(original_file)


# 超参数
batch_size = 2

# 获取已经训练好的模型
unet_model = Unet.get_trained_unet()

# 获取需要进行预测的图像
x_predict, file_name_list = mha_data_reader.get_predict_data(env.PREDICT_IMAGES_DIR)

# 进行预测，返回结果类型是Numpy array，返回结果是(x, 512, 512, 1)
result_arr = unet_model.predict(x=x_predict, batch_size=batch_size)

# 遍历预测结果，然后保存
for i in range(len(result_arr)):
    """
    unet模型的最后一层是sigmoid激活函数，所以输入一个像素点，其对应的输出值在[0, 1]
    >=0.5的，我们就认为这个像素是需要分割出来的像素
    """
    # TODO 有没有一个办法，可以把矩阵压缩，然后保存到压缩文件。解压出来之后就是**.mha文件？可以这样吗？
    file_name = file_name_list[i].split(sep='.')[0]
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    save_path = os.path.join(env.PREDICT_OUTPUT_DIR, file_name + ' ' + current_time + '.mha')   # 给文件名加上时间戳
    # 对小数形式的预测值，根据阈值进行划分，1代表是肿瘤，0代表不是
    threshold = 0.5
    result_arr[result_arr >= threshold] = 1
    result_arr[result_arr < threshold] = 1
    img = sitk.GetImageFromArray(result_arr)
    sitk.WriteImage(img, save_path) # 保存.mha图像到磁盘上

    # 压缩后保存
    original_file = save_path
    gzip_file = os.path.join(env.PREDICT_OUTPUT_DIR, file_name + ' ' + current_time + '.gz')
    gzip_compress_and_save(original_file, gzip_file)

    # 删除保存在磁盘上的.mha图像
    os.remove(original_file)




