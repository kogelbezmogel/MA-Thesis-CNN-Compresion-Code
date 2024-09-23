import os
import torch as th

import pickle
import sys
import time

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
from Contribution import Contribution
import LayerSchemes as ls

if __name__ == '__main__':
    # cerating dataloaders    
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()
    
    # layers to prune
    layer_pairs = ls.get_layer_pairs_vgg()
    hierarchical_groups = ls.get_hierarchical_groups_vgg_4h_contr()

    attempts = [ i for i in range(2, 3) ]
    # goal_flops_ratios = [0.64, 0.49, 0.36, 0.25, 0.16] # (2, 7)
    goal_flops_ratios = [0.09] # (1, 2)    
    step_reduction_ratio = 0.08
    retrain_epochs = 4 # in normal use should be 5 for prune and 3 for sensitivity analysis
    last_retrain_epochs = 10 # in normal use shold be 15
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/vgg/Contribution_4h'
    print(f'attempts: {attempts}')
    print(f'ratios: {goal_flops_ratios}')
    print(f'step reduction ratio : {step_reduction_ratio}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"last retrain epochs: {last_retrain_epochs} ")
    print(f"layers: {layer_pairs}")
    print(f"hierarchical groups: {hierarchical_groups}")
    print(f"algorithm main folder: {algorithm_folder_path}")
    print('-------------------------------\n')

    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)


    for goal_flops_ratio in goal_flops_ratios:
        ratio_path = os.path.join(algorithm_folder_path, f'VGG_rat_{int(100 - goal_flops_ratio*100):02d}')
        if not os.path.isdir(ratio_path):
            os.mkdir(ratio_path)

        for attempt in attempts:
            attempt_start = time.time()
            model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/vgg/FineTuned/AN_att{attempt}')
            test_acc = thh.evaluate_model(model, test_dataloader)
            print(f"starting test accuracy: {test_acc:7.4f}")

            alg = Contribution(
                goal_flops_ratio,
                model,
                layer_pairs,
                train_dataloader,
                test_dataloader,
                hierarchical_groups,
                ls.LayerPairs(),
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
