import time

from model import Unet
import data_reader
import SimpleITK as sitk
import env
import os

# 一些参数
batch_size = 2

# 获取已经训练好的模型
unet_model = Unet.get_trained_unet()

# 获取需要进行预测的图像
x_predict, file_name_list = data_reader.get_predict_data(env.PREDICT_IMAGES_DIR)

# 进行预测，返回结果类型是Numpy array，返回结果是(x, 512, 512, 1)
result_arr = unet_model.predict(x=x_predict, batch_size=batch_size)

# 遍历预测结果，另存为图像
for i in range(len(result_arr)):
    file_name = file_name_list[i].split(sep='.')[0]
    current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    save_path = os.path.join(env.PREDICT_OUTPUT_DIR, file_name + ' ' + current_time + '.zip')
    img = sitk.GetImageFromArray(result_arr)
    sitk.WriteImage(img, save_path)

