import os

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

TEMP_BEST_MODEL_FOLDER_SAVE_PATH = os.path.join(BASE_PATH, 'models/temp/')
CIFAR_DATASET_PATH = os.path.join(BASE_PATH, 'dataset/')

RESNET50_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/resnet/resnet_pretrained')
VGG16_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/vgg/vgg_pretrained')
ALEXNET_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/alexnet/alexnet_pretrained')
DENSENET121_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/densenet/densenet_pretrained')
GOOGLENET_ORIGIN_MODEL_PATH = os.path.join(BASE_PATH, 'models/googlenet/googlenet_pretrained')