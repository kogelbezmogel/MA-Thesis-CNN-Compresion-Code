import os
import torch as th

import pickle
import sys
import time

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls
from MeanGradient import MeanGradient


if __name__ == '__main__':
    # creating dataloaders    
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_alexnet()
    # creating hierarchical groups based on sensitivity analysis
    hierarchical_groups = ls.get_hierarchical_groups_alexnet_2h_v3_mean()

    attempts = [ i for i in range(2, 3) ]
    # goal_flops_ratios = [0.82, 0.66, 0.51, 0.39, 0.28, 0.19, 0.11, 0.06]
    goal_flops_ratios = [0.19, 0.11, 0.06]
    
    retrain_epochs = 4 # 5 is in the paper
    last_retrain_epochs = 10 # in normal use shold be 10
    n = 64
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/alexnet/MeanGradient_2h_064_v3'
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
        ratio_path = os.path.join(algorithm_folder_path, f'AlexNet_rat_{int(100 - goal_flops_ratio*100):02d}')
        if not os.path.isdir(ratio_path):
            os.mkdir(ratio_path)

        for attempt in attempts:
            attempt_start = time.time()
            model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/alexnet/FineTuned/AN_att{attempt}')
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

# Time consumption notes
# h2 64 (1-9)r -> 
