"""训练模型
"""
import data_reader
import Unet
import env
from keras.callbacks import ModelCheckpoint


# 一些超参数
batch_size = 2
epochs = 10

# 获取数据
x_train, y_train = data_reader.getTrainData()

# TODO 进行归一化，归一化到区间[0, 1]
x_train = x_train / 255.0
y_train = y_train / 255.0

# 获取模型
unet_model = Unet.get_untrained_unet()

# 设置模型相关参数
unet_model.compile(optimizer='Adam', loss='cce_dice_loss', metrics=['dice_score', 'iou_score', 'jaccard_score', 'f1_score', 'f2_score'])
unet_model.summary()    # 打印到控制台看一下模型结构

checkpoint = ModelCheckpoint(filepath=env.MODEL_SAVE_DIR, monitor='val_score', verbose=1, save_best_only=True, mode='max')

# TODO 开始训练模型
"""
训练时要对数据进行分块，拿一部分出来作为验证集
训练完成后模型会自动保存到指定的路径中
"""
history = unet_model.fit(x=x_train, y=y_train, batch_size=batch_size, callbacks=[checkpoint], epochs=epochs, validation_split=0.2, shuffle=False)
