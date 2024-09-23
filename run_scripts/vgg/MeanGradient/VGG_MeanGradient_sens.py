import os
import torch as th

import pickle
import sys

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls
from MeanGradient import MeanGradient


if __name__ == '__main__':
    # creatgin dataloaders
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_vgg()
    # creating hierarchical groups based on sensitivity analysis
    hierarchical_groups = ls.get_hierarchical_groups_vgg_1h()

    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/vgg/MeanGradient_sens'
    static_group = []

    attempts = [ i for i in range(2, 5) ]
    retrain_epochs = 0 # in normal use should be 5 for prune and 3 for sensitivity analysis
    n = 256
    print(f'attempts: {attempts}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"n: {n}")
    print(f"layers: {layer_pairs}")
    print(f"hierarchical groups: {hierarchical_groups}")
    print(f"algorithm main folder: {algorithm_folder_path}")
    print('-------------------------------\n')


    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)

    for attempt in attempts:
        model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/vgg/FineTuned/AN_att{attempt}')

        alg = MeanGradient(
            0.1,
            model,
            layer_pairs,
            train_dataloader,
            test_dataloader,
            hierarchical_groups,
            static_group,
            retrain_epochs=retrain_epochs,
            n=n,
            flops_constraint=True
        )
        alg.sensitivity_analysis(portion_size=0.1)
        results = alg.sensitivity_results

        attempt_model_analysis_path = os.path.join(algorithm_folder_path, f'VG_at{attempt}_analysis.pickle')
        pickle.dump(results, open(attempt_model_analysis_path, 'wb'))
        print( f'attempt: {attempt} done | dir: {algorithm_folder_path}' )
