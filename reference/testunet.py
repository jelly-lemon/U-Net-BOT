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

def main():

    parser = argparse.ArgumentParser(description="Comline for test")
    parser.add_argument('-weight',type=str,required=True)   
    parser.add_argument('-gpus',type=str,default='0')
    parser.add_argument('-batchsize',type=int,default=8)
    parser.add_argument('-loss',type=str,default='cce_dice_loss')
    parser.add_argument('-threshold',type=float,default=0.7)
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
    model.load_weights(os.path.join('../weights',model_name, backbone, loss_func, args.weight))
    model.compile('Adam', loss=args.loss, metrics=['dice_score','iou_score','jaccard_score','f1_score','f2_score'])
    model.summary() 


    te_set = os.listdir(os.path.join('../MHA_images','new_masks','filtered','test'))
    mha_path = os.path.join('../predict','MHA_masks', model_name, backbone, loss_func, args.weight.split('.h5')[0])
    if not os.path.exists(mha_path):
        os.makedirs(mha_path)

    te_list = []
    loss_list = []
    dice_score_list = []
    iou_score_list = []
    jac_score_list = []
    f1_score_list = []
    f2_score_list = []
    for mha_file in te_set:
        x_te_path = os.path.join('../MHA_images','images','test',mha_file)
        y_te_path = os.path.join('../MHA_images','new_masks','filtered','test',mha_file)
        te_id = mha_file.split('.')[0]
        te_list.append(te_id)
        # load data
        x_te = sitk.ReadImage(x_te_path)
        x_te_arr = sitk.GetArrayFromImage(x_te)
        y_te = sitk.ReadImage(y_te_path)
        y_te_arr = sitk.GetArrayFromImage(y_te)

        x_te_pr = x_te_arr[:,:,:,np.newaxis]
        print(x_te_pr.shape)
        y_te_pr = y_te_arr[:,:,:,np.newaxis]
        print(y_te_pr.shape)
        # Normalize to [-1.0, 1.0] interval (expected by model)
        x_te_pr = (2.0 / 255.0) * x_te_pr - 1.0

        # evaluate
        print(model.metrics_names)
        evaluate = model.evaluate(
            x_te_pr, 
            y_te_pr,
            verbose=1,
            batch_size=args.batchsize
        )
        print(te_id)
        print(evaluate)
        loss_list.append(evaluate[0])
        dice_score_list.append(evaluate[1])
        iou_score_list.append(evaluate[2])
        jac_score_list.append(evaluate[3])
        f1_score_list.append(evaluate[4])
        f2_score_list.append(evaluate[5])
        # predict 
        y_pre_arr = model.predict(x_te_pr,batch_size=args.batchsize)
        save_mha_mask(y_pre_arr,args.threshold,mha_path,te_id)

    # save csv
    csv_name = '%s.csv'%(args.weight.split('.h5')[0])
    test_csv_path = os.path.join('../predict/csv', model_name, backbone, loss_func)
    if not os.path.exists(test_csv_path):
        os.makedirs(test_csv_path)
    csv_path = os.path.join(test_csv_path,csv_name)
    loss_list = ['%.5f'%i for i in loss_list]
    dice_score_list = ['%.5f'%i for i in dice_score_list]
    iou_score_list = ['%.5f'%i for i in iou_score_list]
    jac_score_list = ['%.5f'%i for i in jac_score_list]
    f1_score_list = ['%.5f'%i for i in f1_score_list]
    f2_score_list = ['%.5f'%i for i in f2_score_list]

    data = {'patient_id':te_list,'loss':loss_list,'dice_score':dice_score_list,'iou_score':iou_score_list,'jac_score':jac_score_list,'f1_score':f1_score_list
            ,'f2_score':f2_score_list}
    frame = pd.DataFrame(data)
    frame.to_csv(csv_path,index=0)


    
if __name__ == '__main__':
    main()

