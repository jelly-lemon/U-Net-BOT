import SimpleITK as sitk
import threading
import numpy as np
from tqdm import tqdm
import os

window_center = 40
window_width = 400
win_min = (2 * window_center - window_width) / 2 + 0.5
win_max = (2 * window_center + window_width) / 2 + 0.5
dFactor = 255.0 / (win_max - win_min)


class ConvertThread(threading.Thread):
    def __init__(self, name, MHA_list):
        threading.Thread.__init__(self)
        self.name = name
        self.MHA_list = MHA_list

    def run(self):
        convert_MHA(self.name, self.MHA_list)


def convert(path, new_path):
    if path is None:
        return

    MHA = sitk.ReadImage(path)
    MHA_array = sitk.GetArrayFromImage(MHA)
    shape = MHA_array.shape
    MHA_array = MHA_array.ravel()
    num_pix = MHA_array.shape[0]
    new_MHA_array = np.zeros(MHA_array.shape, dtype="float32")

    for i in range(num_pix):
        pixel_val = MHA_array[i]
        if pixel_val < win_min:
            new_MHA_array[i] = 0
            continue
        if pixel_val > win_max:
            new_MHA_array[i] = 255
            continue

        pixel_val = (pixel_val - win_min) * dFactor
        if pixel_val < 0:
            new_MHA_array[i] = 0
        elif pixel_val > 255:
            new_MHA_array[i] = 255
        else:
            new_MHA_array[i] = pixel_val

    new_MHA_array = new_MHA_array.reshape(shape)
    new_MHA = sitk.GetImageFromArray(new_MHA_array)
    sitk.WriteImage(new_MHA, new_path, True)


def convert_MHA(threadName, list):
    MHA_path = "./original_MHA"
    new_MHA_path = "./new_MHA"

    MHA_list = os.listdir(MHA_path)

    pbar = tqdm(list)
    pbar.set_description(threadName + ": ")
    for path in pbar:
        convert(os.path.join(MHA_path, path), os.path.join(new_MHA_path, path))


def chop(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]


def main():
    MHA_path = "./original_MHA"
    MHA_list = os.listdir(MHA_path)
    chop_list = chop(MHA_list, 6)
    id = 1

    for i in chop_list:
        ConvertThread("Thread-"+str(id), i).start()
        id = id + 1


if __name__ == "__main__":
    main()
