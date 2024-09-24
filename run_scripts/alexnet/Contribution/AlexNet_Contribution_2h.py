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
from Contribution import Contribution
import LayerSchemes as ls

if __name__ == '__main__':

    if not os.path.isfile(config.ALEXNET_ORIGIN_MODEL_PATH):
        model = thv.models.alexnet(weights='IMAGENET1K_V1')
        th.save(model, config.ALEXNET_ORIGIN_MODEL_PATH)

    # cerating dataloaders    
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()
    
    # layers to prune
    layer_pairs = ls.get_layer_pairs_alexnet()
    hierarchical_groups = ls.get_hierarchical_groups_alexnet_2h_contr()
    static_group = []

    attempts = [ i for i in range(0, 3) ]
    goal_flops_ratios = [0.82, 0.66, 0.51, 0.39, 0.28, 0.19, 0.11, 0.06]

    step_reduction_ratio = 0.07
    retrain_epochs = 4
    last_retrain_epochs = 10
    algorithm_folder_path = os.path.join(config.BASE_PATH, 'results/alexnet/Contribution_2h_007')
    print(f'attempts: {attempts}')
    print(f'ratios: {goal_flops_ratios}')
    print(f'step reduction ratio : {step_reduction_ratio}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"last retrain epochs: {last_retrain_epochs} ")
    print(f"layers: {layer_pairs}")
    print(f"hierarchical groups: {hierarchical_groups}")
    print(f"static group: {static_group}")
    print(f"algorithm main folder: {algorithm_folder_path}")
    print('-------------------------------\n')

    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)


    for goal_flops_ratio in goal_flops_ratios:
        ratio_path = os.path.join(algorithm_folder_path, f'AlexNet_rat_{int(100 - goal_flops_ratio*100):02d}')
        if not os.path.isdir(ratio_path):
            os.mkdir(ratio_path)

        for attempt in attempts:
            attempt_start = time.time()
            model = th.load(os.path.join(config.BASE_PATH, f'models/finetuned/alexnet/AN_att{attempt}'))
            test_acc = thh.evaluate_model(model, test_dataloader)
            print(f"starting test accuracy: {test_acc:7.4f}")

            alg = Contribution(
                goal_flops_ratio,
                model,
                layer_pairs,
                train_dataloader,
                test_dataloader,
                hierarchical_groups,
                static_group,
                retrain_epochs,
                pruning_sample_size=256,
                step_reduction_ratio=step_reduction_ratio,
                flops_constraint=True
                )
            alg.prune_model()

            history = thh.train_model(model, train_dataloader, test_dataloader, epochs=last_retrain_epochs)
            
            attempt_model_path = os.path.join(ratio_path, f'AN_at{attempt}')
            attempt_history_path = os.path.join(ratio_path, f'AN_at{attempt}_history.pickle')
            pickle.dump(history, open(attempt_history_path, 'wb'))
            th.save(model, attempt_model_path)
            attempt_end = time.time()
            print( f'attempt: {attempt} | time: {round(attempt_end-attempt_start, 1)}s dir: {attempt_model_path}' )