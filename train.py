import reader
import env
from Unet import Unet
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras import regularizers
from keras import backend as K


def learning_rate_scheduler(epoch, model):
    if epoch != 0 and epoch % 5 == 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr / 10)

    return K.get_value(model.optimizer.lr)

def main():
    # TODO 读取训练数据
    """
    先进入人工标注文件夹 -> 看有哪些已经标记好了的CT图像，读入所有数据 ->
    根据人工标记好了的CT图像名称去读取原始CT图像 -> 读入原始CT图像 -> 把训练数据组合成一个 tensor
    """
    # image_array, mask_array = reader.read_file_2_array([env.TRAIN_IMAGES_DIR, env.TRAIN_PROCESSED_MASKS_DIR])


    # TODO 开始训练
    """
    训练时要对数据进行分块，拿一部分出来作为验证集
    """
    model = Unet()
    lr_scheduler = LearningRateScheduler(learning_rate_scheduler)


    # TODO 保存模型


if __name__ == "__main__":
    main()
