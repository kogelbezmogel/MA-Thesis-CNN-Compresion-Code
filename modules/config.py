import os

BASE_PATH = os.path.abspath( os.path.join(os.path.dirname(os.path.abspath(__file__)), '..') ) 

TEMP_BEST_MODEL_FOLDER_SAVE_PATH = os.path.join(BASE_PATH, 'models/temp/')
CIFAR_DATASET_PATH = os.path.join(BASE_PATH, 'data/dataset/')

RESNET50_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/pretrained/resnet_pretrained')
VGG16_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/pretrained/vgg_pretrained')
ALEXNET_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/pretrained/alexnet_pretrained')
DENSENET121_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/pretrained/densenet_pretrained')
GOOGLENET_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/pretrained/googlenet_pretrained')