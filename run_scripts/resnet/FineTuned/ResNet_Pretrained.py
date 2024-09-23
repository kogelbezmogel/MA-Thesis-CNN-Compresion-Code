import torch as th
import torchvision as thv
import pickle
import os
import sys
sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh

pretrained_models_folder_path = '/net/tscratch/people/plgkogel/results/resnet/FineTuned/'

if __name__ == '__main__':

    if not os.path.isfile(thh.config.resnet50_origin_model_path):
        model = thv.models.alexnet(weights='IMAGENET1K_V1')
        th.save(model, thh.config.resnet50_origin_model_path)

    attempts  = 3
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    for i in range(0, attempts):
        model = thh.ResNet50T()
        history = thh.finetune_model(model, train_dataloader, test_dataloader, epochs_pair=(10, 30), fc_name='fc')

        model_name = f'AN_att{i}'
        history_name = f'AN_hist{i}.pickle'
        full_model_path = os.path.join(pretrained_models_folder_path, model_name)
        full_history_path = os.path.join(pretrained_models_folder_path, history_name)

        th.save(model, full_model_path)
        pickle.dump(history, open(full_history_path, 'wb'))
        print(f'attempt: {i} done ------------------\n')
