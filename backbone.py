import keras.applications.vgg19
import keras.applications.resnet50
from keras.layers import Input, Conv2D


def get_backbone(backbone="resnet50", input=None, input_size=(512, 512), weights="imagenet", include_top=True,
                 pooling=None):
    if input is None:
        input = Input(shape=input_size)

    if backbone is None:
        return input
    elif backbone == "resnet50":
        return Resnet50(input=input, input_size=input_size, weights=weights, include_top=include_top, pooling=pooling)


def Resnet50(input=None, input_size=(512, 512), weights="imagenet", pooling=None, include_top=True):
    input = Conv2D(filters=3, kernel_size=1, activation="relu")(input)
    print(input)
    res = keras.applications.resnet50.ResNet50(include_top=include_top, input_shape=(input_size[0], input_size[1], 3),
                                               input_tensor=input, weights=weights, pooling=pooling)
    res = Conv2D(filters=1, kernel_size=1, activation="relu")(res)
    return res
