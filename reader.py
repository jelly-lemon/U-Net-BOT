# coding: utf-8
import os
import threading
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np


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
    file_list = os.listdir(path)
    file_list.sort()

    pbar = tqdm(file_list[0:3])
    pbar.set_description(description)
    array_list = None
    for file in pbar:
        array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(path, file)))

        if array_list is None:
            array_list = array[:, :, :, np.newaxis]
        else:
            array_list = np.concatenate([array_list, array[:, :, :, np.newaxis]], axis=0)

    return array_list


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