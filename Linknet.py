from segmentation_models import Linknet


def linknet(backbone_name='vgg16',
            input_shape=(None, None, 3),
            classes=1,
            activation='sigmoid',
            encoder_weights='imagenet',
            encoder_freeze=False,
            encoder_features='default',
            decoder_filters=(None, None, None, None, 16), 
            decoder_use_batchnorm=True,
            decoder_block_type='upsampling',
            **kwargs):
    return Linknet(backbone_name, input_shape, classes, activation, encoder_weights, encoder_freeze, encoder_features,
                   decoder_filters, decoder_use_batchnorm, decoder_block_type, **kwargs)
