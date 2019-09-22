# coding: utf-8
"""用来获取训练数据、验证数据
"""
import os
import threading
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np

# TODO 获取待预测数据
def get_predict_data():
    return





class ReadThread(threading.Thread):
    def __init__(self, description, path):
        threading.Thread.__init__(self)
        self.description = description
        self.path = path
        self.array = None

    def run(self):
        self.array = get_files(self.description, self.path)

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


def get_files(description, path):
    """
    获取指定路径下的文件，并转化为ndarray

    :param description:该路径描述，运行时打印出来给人看的
    :param path:指定的路径
    :return:所有文件组成的ndarray
    """
    file_list = os.listdir(path)
    file_list.sort()

    pbar = tqdm(file_list)
    pbar.set_description(description)
    array_list = None
    for file in pbar:
        array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file)))

        if array_list is None:
            array_list = array[:, :, :, np.newaxis]
        else:
            array_list = np.concatenate([array_list, array[:, :, :, np.newaxis]], axis=0)

    return array_list


def get_train_data(paths=[]):
    """
    利用多线程，将指定路径下的mha文件读取到内存，并转化为ndarray
    一个路径使用一个线程

    :param paths:路径列表
    :return:每个路径下的所有mha文件组成的ndarray
    """
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

    # 一个路径对应一个线程。一个线程会读取对应路径下的所有mha文件并转化为ndarray
    for ta in threads:
        yield ta.get_array()
