import numpy as np
import SimpleITK as sitk
import os
from tqdm import tqdm


mask_path = "./mask"
mha_path = "./new_MHA"
mask_cut_path = "./mask_cut"
mha_cut_path = "./MHA_cut"

mask_list = os.listdir(mask_path)
mha_list = os.listdir(mha_path)

pbar = tqdm(mask_list, unit="file")
pbar.set_description("cutting: ")
for file in pbar:
    mha = file.split("-")[0] + ".mha"
    if mha not in mha_list:
        print("%s is lost" % file)
        continue

    start = int(file.split("-")[1]) - 1
    end = int(file.split("-")[2].split(".")[0]) - 1

    mask_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mask_path, file)))[start: end, :, :]
    mha_array = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(mha_path, mha)))[start: end, :, :]

    sitk.WriteImage(sitk.GetImageFromArray(mask_array), os.path.join(mask_cut_path, "mask_"+mha), True)
    sitk.WriteImage(sitk.GetImageFromArray(mha_array), os.path.join(mha_cut_path, mha), True)
