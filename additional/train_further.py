import pickle
import torch as th
import sys
import os

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'modules'))
sys.path.append( CONFIG_PATH )

import config
import torchhelper as thh
import os

epochs = 10
train_dataloader = thh.get_train_dataloader()
test_dataloader = thh.get_test_dataloader()

main_path = os.path.join(config.BASE_PATH, 'results/resnet/IndirectDirect_wo_steps/')

# ratios = [17, 33, 47, 59, 69, 77, 83, 88] # googlenet
ratios = [13, 25, 36, 46, 55, 64, 71, 78] # resnet
# ratios = [19, 36, 51, 64, 75, 84, 91, 96] # vgg
# ratios = [18, 34, 49, 61, 72, 81, 89, 94] # alexnet

folder_prefix = '_'.join( os.listdir(main_path)[0].split('_')[:-1] )

# creating folder in which files will be saved after further training
save_folder = os.path.join('/'.join( main_path.split("/")[:-2] ) ,f'{main_path.split("/")[-2]}_further')
if not os.path.isdir(save_folder):
    os.mkdir(save_folder)

for ratio in ratios:
    folder_name = folder_prefix + f'_{ratio}'
    folder_path = os.path.join(main_path, folder_name)
    file_prefix = os.listdir(folder_path)[0].split('_')[0]

    for attempt in range(0, 3):
        history_file_name = file_prefix + f"_at{attempt}_history.pickle"
        model_file_name = file_prefix + f"_at{attempt}"
        print(history_file_name)
        print(model_file_name)

        history = pickle.load(open(os.path.join(main_path, folder_name, history_file_name), 'rb'))
        model = th.load( os.path.join(main_path, folder_name, model_file_name) )

        new_history = thh.train_model(model, train_dataloader, test_dataloader, epochs)
        history += new_history

        if not os.path.isdir( os.path.join(save_folder, folder_name) ):
            os.mkdir( os.path.join(save_folder, folder_name) )

        th.save( model, os.path.join(save_folder, folder_name, model_file_name) )
        pickle.dump( history, open(os.path.join(save_folder, folder_name, history_file_name), 'wb') )
    print( f"folder: {folder_name} done" )