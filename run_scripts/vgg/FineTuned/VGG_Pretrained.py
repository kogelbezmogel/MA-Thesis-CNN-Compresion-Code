import torch as th
import torchvision as thv
import pickle
import os
import sys
sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh

pretrained_models_folder_path = '/net/tscratch/people/plgkogel/results/vgg/FineTuned/'

if __name__ == '__main__':

    if not os.path.isfile(thh.config.vgg16_origin_model_path):
        model = thv.models.vgg16(weights='IMAGENET1K_V1')
        th.save(model, thh.config.vgg16_origin_model_path)

    attempts  = 2
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    for i in range(3, 3+attempts):
        model = thh.Vgg16GAP()
        history = thh.finetune_model(model, train_dataloader, test_dataloader, epochs_pair=(10, 20))

        model_name = f'AN_att{i}'
        history_name = f'AN_hist{i}.pickle'
        full_model_path = os.path.join(pretrained_models_folder_path, model_name)
        full_history_path = os.path.join(pretrained_models_folder_path, history_name)

        th.save(model, full_model_path)
        pickle.dump(history, open(full_history_path, 'wb'))
        print(f'attempt: {i} done ------------------\n')
