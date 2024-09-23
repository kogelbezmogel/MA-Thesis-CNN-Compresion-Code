import os
import torch as th

import time
import pickle
import sys

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls
from Random import RandomRat

if __name__ == '__main__':
    attempts = [ i for i in range(0, 3) ]
    goal_flops_ratios = [0.82, 0.66, 0.51, 0.39, 0.28, 0.19, 0.11, 0.06]
    print(f'attempts: {attempts}')
    print(f'ratios: {goal_flops_ratios}')
    print('-------------------------------\n')
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/alexnet/RandomRat'

    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_alexnet()
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

            alg = RandomRat(
                goal_flops_ratio,
                model,
                layer_pairs
            )
            alg.prune_model()

            history = thh.train_model(model, train_dataloader, test_dataloader, epochs=15)
            attempt_model_path = os.path.join(ratio_path, f'AN_at{attempt}')
            attempt_history_path = os.path.join(ratio_path, f'AN_at{attempt}_history.pickle')
            pickle.dump(history, open(attempt_history_path, 'wb'))
            th.save(model, attempt_model_path)
            attempt_end = time.time()
            print( f'attempt: {attempt} | time: {attempt_end-attempt_start}sec dir: {attempt_model_path}' )
