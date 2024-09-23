import os
import torch as th
import calflops
import time
import pickle
import sys

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
from FastThiNet import FastThiNet
import LayerSchemes as ls

if __name__ == '__main__':

    # creating dataloaders
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    samples_per_class = 10
    images_samples_path = f'/net/people/plgrid/plgkogel/scratch/datasets/thinetsamples/cifar10_{samples_per_class}'
    if not os.path.isdir(images_samples_path):
       os.mkdir(images_samples_path)
       thh.choose_dataset_representatives(samples_per_class, train_dataloader, images_samples_path)


    # layers to prune
    layer_pairs = ls.get_layer_pairs_googlenet_experimental()

    # definig additional mask for layers
    additional_ratios = dict()
    for pair in layer_pairs:
        t_layer_name = pair['target_layer']
        additional_ratios[t_layer_name] = 1.0
    # additional_ratios['features.0'] = 0.7 # from sesnivity analysis


    attempts = [ i for i in range(0, 3) ]
    ratios = [ round(val/10, 1) for val in range(2, 3) ]
    retrain_epochs = 4 # in normal use should be 2
    last_retrain_epochs = 10 # in normal use shold be 15
    minimize_err = True
    minimize_mode = 'colective'
    locations_per_image = 10
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/googlenet/FastThiNet_w_mini_4e_exp'
    print(f'attempts: {attempts}')
    print(f'ratios: {ratios}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"last retrain epochs: {last_retrain_epochs}")
    print(f"minimize_err: {minimize_err}")
    print(f"minimize_mode: {minimize_mode}")
    print(f"samples per class: {samples_per_class}")
    print(f"locations per image: {locations_per_image}")
    print(f"main folder: {algorithm_folder_path}")
    print(f"additional ratios: {additional_ratios}")
    print('------------------------------------------------------\n')
    assert len(attempts) <= 5


    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)


    for ratio in ratios:
        general_flops_ratio = None
        for attempt in attempts:
            attempt_start = time.time()
            model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/googlenet/FineTuned/AN_att{attempt}')
            test_acc = thh.evaluate_model(model, test_dataloader)
            print(f"starting test accuracy: {test_acc:7.4f}")

            if general_flops_ratio==None:
                flops_start = calflops.calculate_flops(model, (1, 3, 224, 224), output_as_string=False, print_results=False)[0]

            alg = FastThiNet(
                ratio,
                model,
                layer_pairs,
                train_dataloader,
                test_dataloader,
                images_samples_path,
                locations_per_image = locations_per_image,
                minimize_err = minimize_err,
                minimize_mode = minimize_mode,
                retrain_epochs = retrain_epochs,
                additional_ratios_mask = additional_ratios
            )
            alg.prune_model()

            history = thh.train_model(model, train_dataloader, test_dataloader, epochs=last_retrain_epochs)

            # defining folder in which all models will be saved 
            if general_flops_ratio == None:
                flops_pruned = calflops.calculate_flops(model, (1, 3, 224, 224), output_as_string=False, print_results=False)[0]
                general_flops_ratio = round(flops_pruned/flops_start, 2)
                ratio_path = os.path.join(algorithm_folder_path, f'ResNet_rat_{int(100 - general_flops_ratio*100):02d}')
                if not os.path.isdir(ratio_path):
                    os.mkdir(ratio_path)

            attempt_model_path = os.path.join(ratio_path, f'AN_at{attempt}')
            attempt_history_path = os.path.join(ratio_path, f'AN_at{attempt}_history.pickle')
            pickle.dump(history, open(attempt_history_path, 'wb'))
            th.save(model, attempt_model_path)
            attempt_end = time.time()
            print( f'attempt: {attempt} | time: {attempt_end-attempt_start}sec dir: {attempt_model_path}' )
