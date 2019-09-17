import keras.applications.vgg19
import keras.applications.resnet50
from keras.layers import Input


def get_backbone(backbone="resnet50", input=None, input_size=(512, 512, 1), weight="imagenet", include_top=True,
                 pooling=None):
    if input is None:
        input = Input(shape=input_size)

    if backbone is None:
        return input
    elif backbone.lower() is "resnet50":
        return Resnet50(input=input, input_size=input_size, weight=weight, include_top=include_top, pooling=pooling)


def Resnet50(input=None, input_size=(512, 512, 1), weight="imagenet", pooling=None, include_top=True):
    res = keras.applications.resnet50.ResNet50(include_top=include_top, input_size=input_size, input_tensor=input,
                                               weight=weight, pooling=pooling)

    return res
