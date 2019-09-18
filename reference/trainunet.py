from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import jaccard_loss,bce_jaccard_loss,cce_jaccard_loss,dice_loss,bce_dice_loss,cce_dice_loss
from segmentation_models.metrics import iou_score,dice_score,jaccard_score,f1_score,f2_score
import SimpleITK as sitk
import os
import tensorflow as tf
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
import argparse
import pandas as pd
import sys 
sys.path.append('..')
import plot_csv
from PIL import Image

# __all__ = [
#     'iou_score', 'jaccard_score', 'f1_score', 'f2_score', 'dice_score',
#     'get_f_score', 'get_iou_score', 'get_jaccard_score',
# ]
# __all__ = [
#     'jaccard_loss', 'bce_jaccard_loss', 'cce_jaccard_loss',
#     'dice_loss', 'bce_dice_loss', 'cce_dice_loss',
# ]

def main():
   
    parser = argparse.ArgumentParser(description="Comline for train")   
    parser.add_argument('-gpus',type=str,default='0')
    parser.add_argument('-batchsize',type=int,default=8)
    parser.add_argument('-epoch',type=int,default=30)
    parser.add_argument('-backbone',type=str,default='resnet34')
    parser.add_argument('-loss',type=str,default='cce_dice_loss')
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    #config.gpu_options.per_process_gpu_memory_fraction = 0.1
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))     


    tr_set = os.listdir(os.path.join('../MHA_images','new_masks','filtered','train'))
    val_set = os.listdir(os.path.join('../MHA_images','new_masks','filtered','val'))
    x_train_array=None
    y_train_array=None
    x_val_array=None
    y_val_array=None

    for mha_file in tr_set:
        x_tr_path = os.path.join('../MHA_images','images','train',mha_file)
        y_tr_path = os.path.join('../MHA_images','new_masks','filtered','train',mha_file)
        
        # load your data
        x_train = sitk.ReadImage(x_tr_path)
        x_train_arr = sitk.GetArrayFromImage(x_train)
        y_train = sitk.ReadImage(y_tr_path)
        y_train_arr = sitk.GetArrayFromImage(y_train)

        if x_train_array is None:
            x_train_array=x_train_arr
            y_train_array=y_train_arr
        else:
            x_train_array=np.concatenate((x_train_array,x_train_arr),axis=0)
            y_train_array=np.concatenate((y_train_array,y_train_arr),axis=0)

    for mha_file in val_set:
        x_val_path = os.path.join('../MHA_images','images','val',mha_file)
        y_val_path = os.path.join('../MHA_images','new_masks','filtered','val',mha_file)
        
        # load your data
        x_val = sitk.ReadImage(x_val_path)
        x_val_arr = sitk.GetArrayFromImage(x_val)
        y_val = sitk.ReadImage(y_val_path)
        y_val_arr = sitk.GetArrayFromImage(y_val)

        if x_val_array is None:
            x_val_array=x_val_arr
            y_val_array=y_val_arr
        else:
            x_val_array=np.concatenate((x_val_array,x_val_arr),axis=0)
            y_val_array=np.concatenate((y_val_array,y_val_arr),axis=0)
        

    print(x_train_array.shape)
    print(y_train_array.shape)
    print(x_val_array.shape)
    print(y_val_array.shape)


    # sitk.WriteImage(sitk.GetImageFromArray(x_train_array),'x_train.mha')
    # sitk.WriteImage(sitk.GetImageFromArray(y_train_array),'y_train.mha')
    # sitk.WriteImage(sitk.GetImageFromArray(x_val_array),'x_val.mha')
    # sitk.WriteImage(sitk.GetImageFromArray(y_val_array),'y_val.mha') 

    # shuffle the images
    indices = np.random.permutation(x_train_array.shape[0])
    x_train_array = x_train_array[indices]
    y_train_array = y_train_array[indices]
    indices = np.random.permutation(x_val_array.shape[0])
    x_val_array = x_val_array[indices]
    y_val_array = y_val_array[indices]

    x_train_pr = x_train_array[:,:,:,np.newaxis]
    y_train_pr = y_train_array[:,:,:,np.newaxis]
    x_val_pr = x_val_array[:,:,:,np.newaxis]
    y_val_pr = y_val_array[:,:,:,np.newaxis]

    print(x_train_pr.shape)
    print(y_train_pr.shape)
    print(x_val_pr.shape)
    print(y_val_pr.shape)

    # Normalize to [-1.0, 1.0] interval (expected by model)
    x_train_pr = (2.0 / 255.0) * x_train_pr - 1.0
    x_val_pr = (2.0 / 255.0) * x_val_pr - 1.0   
    # net start
    BACKBONE = args.backbone
    # preprocess_input = get_preprocessing(BACKBONE)

    # define model
    model = Unet(BACKBONE, input_shape=(512, 512, 1), encoder_weights=None)
    model.compile('Adam', loss=args.loss, metrics=['dice_score','iou_score','jaccard_score','f1_score','f2_score'])
    model.summary() 

    # fit model
    unet_model_path = "../weights/unet/%s/%s"%(args.backbone,args.loss)
    if not os.path.exists(unet_model_path):
        os.makedirs(unet_model_path)
    file_path = "%s/unet_%s_%s_{epoch:02d}_{val_score:.5f}.h5"%(unet_model_path,args.backbone,args.loss)
    checkpoint = ModelCheckpoint(file_path, monitor='val_score', verbose=1, save_best_only=True, mode='max')

    history = model.fit(
        x=x_train_pr,
        y=y_train_pr,
        batch_size=args.batchsize,
        callbacks=[checkpoint],
        epochs=args.epoch,
        validation_data=(x_val_pr, y_val_pr),
        shuffle=False
    )

    # save csv
    csv_name = 'unet_%s_%s_epoch%s_batchsize%s.csv'%(args.backbone,args.loss,args.epoch,args.batchsize)
    unet_csv_path = "../record/csv/unet/%s/%s"%(args.backbone,args.loss)
    if not os.path.exists(unet_csv_path):
        os.makedirs(unet_csv_path)
    csv_path = os.path.join(unet_csv_path,csv_name)
    loss_list = ['%.5f'%i for i in history.history['loss']]
    dice_score_list = ['%.5f'%i for i in history.history['score']]
    iou_score_list = ['%.5f'%i for i in history.history['iou_score']]
    jac_score_list = ['%.5f'%i for i in history.history['iou_score_1']]
    f1_score_list = ['%.5f'%i for i in history.history['score_1']]
    f2_score_list = ['%.5f'%i for i in history.history['score_2']]

    val_loss_list = ['%.5f'%i for i in history.history['val_loss']]
    val_dice_score_list = ['%.5f'%i for i in history.history['val_score']]
    val_iou_score_list = ['%.5f'%i for i in history.history['val_iou_score']]
    val_jac_score_list = ['%.5f'%i for i in history.history['val_iou_score_1']]
    val_f1_score_list = ['%.5f'%i for i in history.history['val_score_1']]
    val_f2_score_list = ['%.5f'%i for i in history.history['val_score_2']]

    data = {'loss':loss_list,'dice_score':dice_score_list,'iou_score':iou_score_list,'jac_score':jac_score_list,'f1_score':f1_score_list
            ,'f2_score':f2_score_list,'val_loss':val_loss_list,'val_dice_score':val_dice_score_list,'val_iou_score':val_iou_score_list
            ,'val_jac_score':val_jac_score_list,'val_f1_score':val_f1_score_list,'val_f2_score':val_f2_score_list}
    frame = pd.DataFrame(data)
    frame.to_csv(csv_path,index=0)

    # plot and save image
    unet_image_path = "../record/images/unet/%s/%s"%(args.backbone,args.loss)
    save_image_name = 'unet_%s_%s_epoch%s_batchsize%s.png'%(args.backbone,args.loss,args.epoch,args.batchsize)
    if not os.path.exists(unet_image_path):
        os.makedirs(unet_image_path)    
    save_image_path = os.path.join(unet_image_path,save_image_name)
    plot_csv.plot_and_save(csv_path,save_image_path)
    
if __name__ == '__main__':
    main()

