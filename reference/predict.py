from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import jaccard_loss,bce_jaccard_loss,cce_jaccard_loss,dice_loss,bce_dice_loss,cce_dice_loss
from segmentation_models.metrics import iou_score,dice_score,jaccard_score,f1_score,f2_score
import os
import tensorflow as tf
import SimpleITK as sitk
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import argparse
import pandas as pd
from PIL import Image
import cv2

# __all__ = [
#     'iou_score', 'jaccard_score', 'f1_score', 'f2_score', 'dice_score',
#     'get_f_score', 'get_iou_score', 'get_jaccard_score',
# ]
# __all__ = [
#     'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
#     'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
# ]

def save_mha_mask(pre_arr,threshold,save_path,file_name):
    shape = pre_arr.shape
    mha_arr = pre_arr.reshape(shape[0],shape[1],shape[2])

    mha_arr[mha_arr>=threshold] = 1
    mha_arr[mha_arr<threshold] = 0

    sitk.WriteImage(sitk.GetImageFromArray(mha_arr),os.path.join(save_path,file_name+'.mha')) 

def get_msgs(file_name):
    msgs_list = file_name.split('.')[0].split('_')
    model_name = msgs_list[0]
    backbone = msgs_list[1]
    loss_func = ''
    for i in range(2,msgs_list.index('loss')):
        loss_func += msgs_list[i]+'_'
    loss_func += 'loss'

    return model_name, backbone, loss_func

def load_png_files(image_path,start,end):
    im_array = []

    for i in range(start,end+1):
        im = Image.open(os.path.join(image_path,str(i)+'.png'))
        tmp_array = np.array(im)
        tmp_array = tmp_array[np.newaxis,:,:]
        if len(im_array) == 0:
            im_array = tmp_array
        else:
            im_array = np.concatenate((im_array,tmp_array),axis=0)
    return im_array

def main():

    parser = argparse.ArgumentParser(description="Comline for test")
    parser.add_argument('-weight',type=str,required=True)   
    parser.add_argument('-gpus',type=str,default='0')
    parser.add_argument('-batchsize',type=int,default=8)
    parser.add_argument('-loss',type=str,default='cce_dice_loss')
    parser.add_argument('-threshold',type=float,default=0.7)
    parser.add_argument('-patient',type=str)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))    

    model_name, backbone, loss_func = get_msgs(args.weight)

    # net start
    BACKBONE = backbone

    # define model
    model = Unet(BACKBONE, input_shape=(512, 512, 1), encoder_weights=None)
    model.load_weights(os.path.join('./weights',model_name, backbone, loss_func, args.weight))
    model.compile('Adam', loss=args.loss, metrics=['dice_score','iou_score','jaccard_score','f1_score','f2_score'])
    model.summary() 

    # load data
    patient_path = os.path.join('PNG_images',args.patient)
    files_count = len(os.listdir(patient_path))
    x_te_arr = load_png_files(patient_path,1,files_count)

    x_te_pr = x_te_arr[:,:,:,np.newaxis]
    print(x_te_pr.shape)

    # Normalize to [-1.0, 1.0] interval (expected by model)
    x_te_pr = (2.0 / 255.0) * x_te_pr - 1.0

    # predict 
    mha_path = './'
    y_pre_arr = model.predict(x_te_pr,batch_size=args.batchsize)
    save_mha_mask(y_pre_arr,args.threshold,mha_path,args.patient)



    
if __name__ == '__main__':
    main()

