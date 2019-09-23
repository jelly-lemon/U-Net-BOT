# coding: utf-8
"""用来获取训练数据、验证数据
"""
import os
import threading
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

# TODO 获取待预测数据
def getPredictData():
    return


# TODO 读取训练数据
def getTrainData():
    """先进入人工标注文件夹 -> 看有哪些已经标记好了的CT图像，读入所有数据 ->
    根据人工标记好了的CT图像名称去读取原始CT图像 -> 读入原始CT图像 -> 把训练数据组合成一个 tensor

    """
    return


class ReadThread(threading.Thread):
    def __init__(self, description, path):
        threading.Thread.__init__(self)
        self.description = description
        self.path = path
        self.array = None

    def run(self):
        file_list = os.listdir(self.path)
        file_list.sort()

        pbar = tqdm(file_list)
        pbar.set_description(self.description)
        array_list = None
        for file in pbar:
            array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.path, file)))

            if array_list is None:
                array_list = array[:, :, :, np.newaxis]
            else:
                array_list = np.concatenate([array_list, array[:, :, :, np.newaxis]], axis=0)

        self.array = array_list

    def get_array(self):
        try:
            return self.array

        except Exception:
            return None

    def get_path(self):
        try:
            return self.path

        except Exception:
            return None


def read_file_2_array(paths=[]):
    # 线程数组
    threads = []

    # 根据路径创建线程
    for path in paths:
        thread = ReadThread("Reading file from " + path, path)
        threads.append(thread)
        thread.start()

    # 等待所有线程结束
    for t in threads:
        t.join()

    print("All file have been read to memory!")

    for ta in threads:
        yield ta.get_array()
