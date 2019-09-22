import Unet
import data_reader
import SimpleITK as sitk

# 一些参数
batch_size = 1

# 获取已经训练好的模型
unet_model = Unet.get_trained_unet()

# 获取需要进行预测的图像
x_predict = data_reader.get_predict_data()

# 进行预测，返回结果类型是Numpy array
result_arr = unet_model.predict(x=x_predict, batch_size=batch_size)
sitk.GetImageFromArray()

# TODO 将预测结果转化为图片
