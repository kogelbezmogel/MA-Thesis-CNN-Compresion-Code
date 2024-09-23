import os
import torch as th
import pickle
import sys
import time

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls
from FastThiNet import FastThiNet

if __name__ == '__main__':

    # creating dataloaders    
    train_dataloader_single = thh.get_train_dataloader(1)
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()
    samples_per_class = 10
    images_samples_path = f'/net/people/plgrid/plgkogel/scratch/datasets/thinetsamples/cifar10_{samples_per_class}'
    if not os.path.isdir(images_samples_path):
       os.mkdir(images_samples_path)
       thh.choose_dataset_representatives(samples_per_class, train_dataloader_single, images_samples_path)

    # layers to prune
    layer_pairs = ls.get_layer_pairs_alexnet()

    # definig additional mask for layers
    additional_ratios = dict()
    for pair in layer_pairs:
        additional_ratios[ pair['target_layer'] ] = 1.0
    
    attempts = [ i for i in range(0, 3) ]
    retrain_epochs = 0
    minimize_err = False
    locations_per_image = 10
    algorithm_folder_path = '/net/people/plgrid/plgkogel/scratch/results/alexnet/FastThiNet_sens'
    print(f'attempts: {attempts}')
    print(f"retrain epochs: {retrain_epochs}")
    print(f"minimize_err: {minimize_err}")
    print(f"samples per class: {samples_per_class}")
    print(f"locations per image: {locations_per_image}")
    print(f"main folder: {algorithm_folder_path}")
    print(f"additional ratios: {additional_ratios}")
    print('------------------------------------------------------\n')


    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)

    for attempt in attempts:
        time_start = time.time()
        model = th.load(f'/net/people/plgrid/plgkogel/scratch/results/alexnet/FineTuned/AN_att{attempt}')
    
        alg = FastThiNet(
            0.1,
            model,
            layer_pairs, 
            train_dataloader,
            test_dataloader,
            images_samples_path,
            locations_per_image=locations_per_image,
            minimize_err=minimize_err,
            retrain_epochs=retrain_epochs,
            additional_ratios_mask=additional_ratios
        )
        alg.sensitivity_analysis(portion_size=0.1)
        results = alg.sensitivity_results

        attempt_model_analysis_path = os.path.join(algorithm_folder_path, f'AN_at{attempt}_analysis.pickle')
        pickle.dump(results, open(attempt_model_analysis_path, 'wb'))
        time_end = time.time()
        print( f'attempt: {attempt} done {round(time_end-time_start, 1)}s | dir: {algorithm_folder_path}' )