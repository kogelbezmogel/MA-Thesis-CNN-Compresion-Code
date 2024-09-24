import os
import torch as th

import pickle
import sys
import time
import torchvision as thv

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'modules'))
sys.path.append( CONFIG_PATH )

import config
import torchhelper as thh
import LayerSchemes as ls
from MeanGradient import MeanGradient


if __name__ == '__main__':

    if not os.path.isfile(config.GOOGLENET_ORIGIN_MODEL_PATH):
        model = thv.models.googlenet(weights='IMAGENET1K_V1')
        th.save(model, config.GOOGLENET_ORIGIN_MODEL_PATH)

    # creating dataloaders    
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_googlenet_classic_v2()
    # creating hierarchical groups based on sensitivity analysis
    hierarchical_groups = ls.get_hierarchical_groups_3h_googlenet_classic()

    attempts = [ i for i in range(0, 3) ]
    goal_flops_ratios = [ 0.92, 0.85, 0.77, 0.69, 0.62, 0.54, 0.46, 0.38 ] # std
    # goal_flops_ratios = [ 0.83, 0.67, 0.53, 0.41, 0.31, 0.23, 0.17, 0.12 ] # exp

    retrain_epochs = 4
    last_retrain_epochs = 10

    n = 75

    algorithm_folder_path = os.path.join(config.BASE_PATH, 'results/googlenet/MeanGradient_3h_std')
    print(f'attempts: {attempts}')
    print(f'ratios: {goal_flops_ratios}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"last retrain epochs: {last_retrain_epochs}")
    print(f"n: {n}")
    print(f"layers: {layer_pairs}")
    print(f"hierarchical groups: {hierarchical_groups}")
    print(f"main folder: {algorithm_folder_path}")
    print("-------------------------------\n")


    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)

    for goal_flops_ratio in goal_flops_ratios:
        ratio_path = os.path.join(algorithm_folder_path, f'GoogLeNet_rat_{int(100 - goal_flops_ratio*100):02d}')
        if not os.path.isdir(ratio_path):
            os.mkdir(ratio_path)

        for attempt in attempts:
            attempt_start = time.time()
            model = th.load( os.path.join(config.BASE_PATH, f'models/finetuned/googlenet/AN_att{attempt}') )
            test_acc = thh.evaluate_model(model, test_dataloader)
            print(f"starting test accuracy: {test_acc:7.4f}")

            alg = MeanGradient(
                goal_flops_ratio,
                model,
                layer_pairs,
                train_dataloader,
                test_dataloader,
                hierarchical_groups,
                static_group=[],
                retrain_epochs=retrain_epochs,
                flops_constraint=False,
                n=n
            )
            alg.prune_model()

            history = thh.train_model(model, train_dataloader, test_dataloader, epochs=last_retrain_epochs)
            
            attempt_model_path = os.path.join(ratio_path, f'AN_at{attempt}')
            attempt_history_path = os.path.join(ratio_path, f'AN_at{attempt}_history.pickle')
            pickle.dump(history, open(attempt_history_path, 'wb'))
            th.save(model, attempt_model_path)
            attempt_end = time.time()
            print( f'attempt: {attempt} | time: {round(attempt_end-attempt_start, 1)}s dir: {attempt_model_path}' )
