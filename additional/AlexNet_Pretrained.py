import torch as th
import torchvision as thv
import pickle
import os
import sys

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'modules'))
sys.path.append( CONFIG_PATH )

import config
import torchhelper as thh

pretrained_models_folder_path = os.path.join(config.BASE_PATH, 'models/finetuned/alexnet')

if __name__ == '__main__':

    if not os.path.isfile(thh.config.ALEXNET_ORIGIN_MODEL_PATH):
        model = thv.models.alexnet(weights='IMAGENET1K_V1')
        th.save(model, thh.config.ALEXNET_ORIGIN_MODEL_PATH)

    attempts = 3
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    for i in range(0, attempts):
        model = thh.AlexNetGAP()
        history = thh.finetune_model(model, train_dataloader, test_dataloader, epochs_pair=(5, 20))

        model_name = f'AN_att{i}'
        history_name = f'AN_hist{i}.pickle'
        full_model_path = os.path.join(pretrained_models_folder_path, model_name)
        full_history_path = os.path.join(pretrained_models_folder_path, history_name)

        th.save(model, full_model_path)
        pickle.dump(history, open(full_history_path, 'wb'))
        print(f'attempt: {i} done ------------------\n')
