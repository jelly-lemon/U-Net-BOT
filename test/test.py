import env
import gzip
import os
import SimpleITK as sitk



def gzip_compress_and_save(original_file: str, gzip_file: str):
    """
    读取一个文件，压缩后另存为.gz文件
    :param original_file:原文件路径
    :param gzip_file:压缩文件路径
    :return:无
    """
    with open(original_file, 'rb') as original_file:
        with gzip.open(gzip_file, 'wb') as zip_file:
            zip_file.writelines(original_file)


gzip_compress_and_save(os.path.join(env.TRAIN_MASKS_DIR, '50655-150-238.mha'),
                       os.path.join(env.TEST_DIR, '50655.gz'))