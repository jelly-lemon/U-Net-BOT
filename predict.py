import Unet
import data_reader

# 一些参数
batch_size = 2

# 获取已经训练好的模型
unet_mode = Unet.get_trained_unet()

# 获取需要进行预测的图像
x_predict = data_reader.getPredictData()

# 进行预测，返回结果类型是Numpy array
result_arr = unet_mode.predict(x=x_predict, batch_size=batch_size)

# TODO 将预测结果转化为图片
