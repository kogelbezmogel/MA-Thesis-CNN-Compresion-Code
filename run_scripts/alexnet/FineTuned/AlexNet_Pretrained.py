import torch as th
import torchvision as thv
import pickle
import os
import sys
sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh

pretrained_models_folder_path = '/net/tscratch/people/plgkogel/results/alexnet/FineTuned/'

if __name__ == '__main__':

    if not os.path.isfile(thh.config.alexnet_origin_model_path):
        model = thv.models.alexnet(weights='IMAGENET1K_V1')
        th.save(model, thh.config.alexnet_origin_model_path)

    attempts = 2
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    for i in range(3, attempts + 3):
        model = thh.AlexNetGAP()
        history = thh.finetune_model(model, train_dataloader, test_dataloader, epochs_pair=(5, 20))

        # model.requires_grad_(False)
        # model.classifier.requires_grad_(True)
        # history_1 = thh.train_model(model, train_dataloader, test_dataloader, epochs=1, lr=0.001)
        # history_2 = thh.train_model(model, train_dataloader, test_dataloader, epochs=3, lr=0.0001)
        # history_3 = thh.train_model(model, train_dataloader, test_dataloader, epochs=6, lr=0.00001)
        # model.requires_grad_(True)
        # history_4 = thh.train_model(model, train_dataloader, test_dataloader, epochs=10, lr=0.0001)
        # history_5 = thh.train_model(model, train_dataloader, test_dataloader, epochs=15, lr=0.00001)
        # history_6 = thh.train_model(model, train_dataloader, test_dataloader, epochs=5, lr=0.000001)
        # full_history = [history_1, history_2, history_3, history_4, history_5]

        model_name = f'AN_att{i}'
        history_name = f'AN_hist{i}.pickle'
        full_model_path = os.path.join(pretrained_models_folder_path, model_name)
        full_history_path = os.path.join(pretrained_models_folder_path, history_name)

        th.save(model, full_model_path)
        pickle.dump(history, open(full_history_path, 'wb'))
        print(f'attempt: {i} done ------------------\n')
