import os
import torch as th

import pickle
import sys
import time 

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls
from Contribution import Contribution

if __name__ == '__main__':
    # cerating dataloaders    
    train_dataloader = thh.get_train_dataloader(128)
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_vgg()
    hierarchical_groups = ls.get_hierarchical_groups_vgg_1h()
    static_group = []

    attempts = [ i for i in range(3, 5) ]
    step_reduction_ratio = 0.0
    retrain_epochs = 0 # in normal use should be 5 for prune and 3 for sensitivity analysis
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/vgg/Contribution_sens1'
    print(f'attempts: {attempts}')
    print(f'step reduction ratio : {step_reduction_ratio}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"layers: {layer_pairs}")
    print(f"hierarchical groups: {hierarchical_groups}")
    print(f"algorithm main folder: {algorithm_folder_path}")
    print('-------------------------------\n')

    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)

    for attempt in attempts:
        time_start = time.time()
        model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/vgg/FineTuned/AN_att{attempt}')
        test_acc = thh.evaluate_model(model, test_dataloader)
        print(f"starting test accuracy: {test_acc:7.4f}")

        alg = Contribution(
            0.1, # goal doesn't matter during sensivity analysis
            model,
            layer_pairs,
            train_dataloader,
            test_dataloader,
            hierarchical_groups,
            static_group,
            retrain_epochs,
            pruning_sample_size=256, # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            step_reduction_ratio=step_reduction_ratio
        )
        alg.sensitivity_analysis(portion_size=0.1) 
        #                                            !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        results = alg.sensitivity_results

        attempt_model_analysis_path = os.path.join(algorithm_folder_path, f'VG_at{attempt}_analysis.pickle')
        pickle.dump(results, open(attempt_model_analysis_path, 'wb'))
        time_end = time.time()
        print( f'attempt: {attempt} done {round(time_end-time_start, 1)}s | dir: {algorithm_folder_path}' )