import os
import torch as th

import time
import pickle
import sys
import calflops
import torchvision as thv

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'modules'))
sys.path.append( CONFIG_PATH )

import config
import torchhelper as thh
import LayerSchemes as ls
from IndirectDirect import IndirectDirect

if __name__ == '__main__':

    if not os.path.isfile(config.VGG16_ORIGIN_MODEL_PATH):
        model = thv.models.vgg16(weights='IMAGENET1K_V1')
        th.save(model, config.VGG16_ORIGIN_MODEL_PATH)

    # creting dataloaders
    train_dataloader = thh.get_train_dataloader()
    test_dataloader = thh.get_test_dataloader()

    # layers to prune
    layer_pairs = ls.get_layer_pairs_vgg()

    attempts = [ i for i in range(0, 3) ]
    ratios = [ round(val/10, 1) for val in range(1, 9) ]
    retrain_epochs = 0
    last_retrain_epochs = 10
    max_reduction = 1.0
    algorithm_folder_path = os.path.join(config.BASE_PATH, 'results/vgg/IndirectDirect_wo_steps')
    print(f'attempts: {attempts}')
    print(f'ratios: {ratios}')
    print(f"max_reduction_in_step: {max_reduction}")
    print(f"retrain epochs: {retrain_epochs}")
    print(f"last retrain epochs: {last_retrain_epochs}")
    print(f"layers: {layer_pairs}")
    print(f"main folder: {algorithm_folder_path}")    
    print('-------------------------------\n')

    if not os.path.isdir(algorithm_folder_path):
        os.mkdir(algorithm_folder_path)

    for ratio in ratios:
        general_flops_ratio = None
        for attempt in attempts:
            attempt_start = time.time()
            model = th.load(os.path.join(config.BASE_PATH, f'models/finetuned/vgg/AN_att{attempt}'))
            test_acc = thh.evaluate_model(model, test_dataloader)
            print(f"starting test accuracy: {test_acc:7.4f}")

            if general_flops_ratio==None:
                flops_start = calflops.calculate_flops(model, (1, 3, 224, 224), output_as_string=False, print_results=False)[0]

            alg = IndirectDirect(
                                    ratio,
                                    model,
                                    layer_pairs,
                                    train_dataloader,
                                    test_dataloader,
                                    retrain_epochs=retrain_epochs,
                                    max_reduction_ratio = max_reduction
                                )
            alg.prune_model()

            history = thh.train_model(model, train_dataloader, test_dataloader, epochs=last_retrain_epochs)

            # defining folder in which all models will be saved 
            if general_flops_ratio == None:
                flops_pruned = calflops.calculate_flops(model, (1, 3, 224, 224), output_as_string=False, print_results=False)[0]
                general_flops_ratio = round(flops_pruned/flops_start, 2)
                ratio_path = os.path.join(algorithm_folder_path, f'VGG_rat_{int(100 - general_flops_ratio*100):02d}')
                if not os.path.isdir(ratio_path):
                    os.mkdir(ratio_path)
  
            attempt_model_path = os.path.join(ratio_path, f'VG_at{attempt}')
            attempt_history_path = os.path.join(ratio_path, f'VG_at{attempt}_history.pickle')
            pickle.dump(history, open(attempt_history_path, 'wb'))
            th.save(model, attempt_model_path)
            attempt_end = time.time()
            print( f'attempt: {attempt} | time: {attempt_end-attempt_start}sec dir: {attempt_model_path}' )
