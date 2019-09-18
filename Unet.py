"""获取unet模型
"""
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from keras.models import Model
from keras.models import load_model


def get_trained_unet():
    model = load_model("*.h5")
    return model


def get_untrained_unet(input_size=(512, 512, 1)):
    """获取unet模型

    # Arguments
        input_size:一张图片的大小，也就是一个输入数据
    """
    input = Input(input_size)  # 返回一个tensor

    conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(input)  # 卷积1
    conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv1)  # 卷积2
    dp1 = Dropout(rate=0.5)(conv2)  # 随机失活1
    mp1 = MaxPooling2D()(dp1)  # 最大池化1

    conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(mp1)  # 卷积3
    conv4 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv3)  # 卷积4
    dp2 = Dropout(rate=0.5)(conv4)  # 随机失活2
    mp2 = MaxPooling2D()(dp2)

    conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(mp2)
    conv6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv5)
    dp3 = Dropout(rate=0.5)(conv6)
    mp3 = MaxPooling2D()(dp3)

    conv7 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(mp3)
    conv8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv7)
    dp4 = Dropout(rate=0.5)(conv8)
    mp4 = MaxPooling2D()(dp4)

    conv9 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation="relu")(mp4)
    conv10 = Conv2D(filters=1024, kernel_size=(3, 3), padding="same", activation="relu")(conv9)
    # dp5 = Dropout(rate=0.5)(conv10)
    up1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(UpSampling2D()(conv10))
    merge1 = concatenate([up1, conv8])

    conv11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(merge1)
    conv12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(conv11)
    # dp6 = Dropout(rate=0.5)(conv12)
    up2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(UpSampling2D()(conv12))
    merge2 = concatenate([up2, conv6])

    conv13 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(merge2)
    conv14 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(conv13)
    # dp7 = Dropout(rate=0.5)(conv14)
    up3 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(UpSampling2D()(conv14))
    merge3 = concatenate([up3, conv4])

    conv15 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(merge3)
    conv16 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(conv15)
    # dp8 = Dropout(rate=0.5)(conv16)
    up4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(UpSampling2D()(conv16))
    merge4 = concatenate([up4, conv2])

    conv17 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(merge4)
    conv18 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(conv17)

    conv19 = Conv2D(filters=2, kernel_size=(3, 3), padding="same", activation="relu")(conv18)

    output = Conv2D(filters=1, kernel_size=(3, 3), padding="same", activation="sigmoid")(conv19)

    model = Model(inputs=input, outputs=output)

    return model
