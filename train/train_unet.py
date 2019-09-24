"""
直接运行该文件训练模型
"""
import data_reader
from model import Unet
import env
from keras.callbacks import ModelCheckpoint
import argparse
import os
import platform
import tensorflow as tf
from keras.backend import tensorflow_backend

# 获取运行时参数
parser = argparse.ArgumentParser(description="train argument")
parser.add_argument("-gpu", type=str, default="0")  # 指定使用哪些GPU
args = parser.parse_args()  # 将获取到的参数字典保存到args变量中

# 运行时使用的GPU
if platform.system() == "Linux":
    # 如果在服务器上运行，就使用GPU
    print("using gpu:", args.gpu)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 设置tensorflow根据需要合理使用显存，而不是直接上来就占满一个GPU的显存
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tensorflow_backend.set_session(tf.Session(config=config))

# 一些超参数
batch_size = 2
epochs = 1

# 获取训练数据
x_train, y_train = data_reader.get_train_data([env.TRAIN_IMAGES_DIR, env.TRAIN_MASKS_DIR])

# 进行归一化，归一化到区间[0, 1]
x_train = x_train / 255.0
y_train = y_train / 255.0

# 获取未训练的模型
unet_model = Unet.get_untrained_unet()

# 设置模型相关参数
# TODO 评估指标用哪些？
unet_model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['acc'])
unet_model.summary()    # 打印到控制台看一下模型结构
checkpoint = ModelCheckpoint(filepath=env.MODEL_SAVE_DIR, monitor='acc', verbose=1, save_best_only=True, mode='max')

# 训练时要对数据进行分块，拿一部分出来作为验证集，训练完成后模型会自动保存到指定的路径中
history = unet_model.fit(x=x_train, y=y_train, batch_size=batch_size, callbacks=[checkpoint],
                         validation_split=0.2, epochs=epochs, shuffle=False)
