import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import os

original_MHA = "./original_MHA"
new_MHA = "./new_MHA"

original_list = os.listdir(original_MHA)
new_list = os.listdir(new_MHA)

pbar = tqdm(original_list)
for file in pbar:
    if file not in new_list:
        print("%s is lost!" % file)
        continue
        
    original = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(original_MHA, file))).shape[0]
    new = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(new_MHA, file))).shape[0]

    if original != new:
        print("%s is wrong!" % file)
