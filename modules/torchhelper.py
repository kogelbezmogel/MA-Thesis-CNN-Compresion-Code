import torch as th
import pandas as pd
import torchvision as thv
import random
import time
import copy
import pickle
import os
import calflops
import config
from typing import Optional, Tuple

def get_random_seed():
    random_data = os.urandom(4)
    random_seed = int.from_bytes(random_data, byteorder="big")
    return random_seed

def set_bn_eval(m):
    if isinstance(m, th.nn.modules.batchnorm._BatchNorm):
        m.eval()


def reset_bn_stats(m):
    if isinstance(m, th.nn.modules.batchnorm._BatchNorm):
        m.reset_running_stats()


def set_bn_train(m):
    if isinstance(m, th.nn.modules.batchnorm._BatchNorm):
        m.train()


def googlenet_loss_train(pred: list, y_org) -> th.Tensor:
    y_pred, y_pred_aux1, y_pred_aux2 = pred
    loss_fun = th.nn.CrossEntropyLoss()
    loss_fun_aux1 = th.nn.CrossEntropyLoss()
    loss_fun_aux2 = th.nn.CrossEntropyLoss()
    return loss_fun(y_pred, y_org) + 0.3*loss_fun_aux1(y_pred_aux1, y_org) + 0.3*loss_fun_aux2(y_pred_aux2, y_org)


def googlenet_loss_evaluate(pred: list, y_org)  -> th.Tensor:
    y_pred, _, _ = pred
    loss_fun = th.nn.CrossEntropyLoss()
    return loss_fun(y_pred, y_org)


def train_model(
                model: th.nn.Module,
                train_dataloader:th.utils.data.DataLoader,
                test_dataloader: th.utils.data.DataLoader,
                epochs: int,
                lr: float=1e-4,
                return_best_test: bool=True,
                return_best_train: bool=False
                ) -> list[dict]:

    learning_history = []
    if return_best_test:
        evaluate_dataloader = test_dataloader
    elif return_best_train:
        evaluate_dataloader = train_dataloader

    if epochs > 0:
        device = th.device("cuda" if th.cuda.is_available() else "cpu")
        model = model.to(device)

        # define name that surely doesn't exists now
        temp_model_name = f'temp_model_{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}'
        temp_best_model_save_path = os.path.join(config.TEMP_BEST_MODEL_FOLDER_SAVE_PATH, temp_model_name)
        while os.path.isfile(temp_best_model_save_path):
            temp_model_name = f'temp_model_{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}{random.randint(100, 999)}'
            temp_best_model_save_path = os.path.join(config.TEMP_BEST_MODEL_FOLDER_SAVE_PATH, temp_model_name)
    
        if return_best_test or return_best_train:
            the_best_acc_score = evaluate_model(model, evaluate_dataloader)
            th.save(model.state_dict(), temp_best_model_save_path)
            print(f"training starts. evaluate_acc: {the_best_acc_score:5.2f}")
        else:
            print(f"training starts. evaluate_acc: {evaluate_model(model, test_dataloader):5.2f}")

        if type(model) == GoogLeNetTAT:
            train_loss_fun = googlenet_loss_train
            test_loss_fun = googlenet_loss_evaluate
        
        else:
            train_loss_fun = th.nn.CrossEntropyLoss()
            test_loss_fun = th.nn.CrossEntropyLoss()
        optimizer = th.optim.Adam( model.parameters(), lr=lr )

        for epoch in range(epochs):
            model.train()
            start = time.time()
            train_accuracy = 0
            test_accuracy = 0
            train_loss = 0
            test_loss = 0

            train_points_num = 0
            for step_train, (x, y_true) in enumerate(train_dataloader):
                # standard learning steps
                x, y_true = x.to(device), y_true.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = train_loss_fun(y_pred, y_true)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()

                # gathering accuracy
                if type(model) == GoogLeNetTAT:
                    y_pred = y_pred[0]
                train_accuracy += (y_pred.argmax(dim=1) == y_true).sum().item()
                train_points_num += x.shape[0]
            train_loss /= (step_train + 1)

            # gathering test accuracy
            with th.no_grad():
                model.eval()
                th.manual_seed(101)
                test_points_num = 0
                for step_test, (x, y_true) in enumerate(test_dataloader):
                    x, y_true = x.to(device), y_true.to(device)
                    y_pred = model(x)
                    loss = test_loss_fun(y_pred, y_true)
                    test_loss += loss.item()
                    if type(model) == GoogLeNetTAT:
                        y_pred = y_pred[0]
                    test_accuracy += (y_pred.argmax(dim=1) == y_true).sum().item()
                    test_points_num += x.shape[0]
                test_loss /= (step_test + 1)
                model.train()
                th.manual_seed( get_random_seed() )
            end = time.time()

            # establishing gathered evaluation metrics
            train_accuracy = train_accuracy / train_points_num * 100
            test_accuracy = test_accuracy / test_points_num * 100

            # checking if the best model has changed
            if return_best_train or return_best_test:
                evaluate_accuracy = evaluate_model(model, evaluate_dataloader)
                if evaluate_accuracy > the_best_acc_score and (return_best_test or return_best_train):
                    the_best_acc_score = evaluate_accuracy
                    th.save(model.state_dict(), temp_best_model_save_path)

            learning_history.append( {'train_loss' : train_loss, 'test_loss' : test_loss, 'train_acc' : train_accuracy, 'test_acc' : test_accuracy} )
            print( f"epoch: {epoch:2d} t: {(end - start):4.0f} sec | train_loss: {train_loss:7.4f} test_loss: {test_loss:7.4f} | train_acc: {train_accuracy:5.2f} test_acc: {test_accuracy:5.2f}")

        # loading the best saved model as a final result
        if return_best_train or return_best_test:
            model.load_state_dict(th.load(temp_best_model_save_path))
            print( f"the best eval_acc: {round(the_best_acc_score, 4)}")

        # removing saved model from temporaries
        if os.path.exists(temp_best_model_save_path):
            os.remove(temp_best_model_save_path)

    return learning_history


def finetune_model(model: th.nn.Module, train_dataloader: th.utils.data.DataLoader, test_dataloader: th.utils.data.DataLoader, epochs_pair: tuple=(5, 10), fc_name: str='classifier') -> list[dict]:

    # freezing features extraction layers
    model.requires_grad_(False)
    model.get_submodule(fc_name).requires_grad_(True)
    if type(model) == GoogLeNetTAT:
        model.aux1.requires_grad_(True)
        model.aux2.requires_grad_(True)
        
    # fine tuning
    history_fin = []
    if epochs_pair[0] > 0:
        print( f"fine tuning" )
        history_fin = train_model(model, train_dataloader, test_dataloader, epochs=epochs_pair[0], lr=1e-4)

    # unfreezing feature extraction layers
    model.requires_grad_(True)

    if epochs_pair[1] > 0:
        print( f"further training")
    history_fur = train_model(model, train_dataloader, test_dataloader, epochs=epochs_pair[1], lr=1e-4)

    return history_fin + history_fur


def evaluate_model(model: th.nn.Module, test_dataloader:th.utils.data.DataLoader) -> float:

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_accuracy = 0
    test_points_num = 0

    th.manual_seed(101)
    with th.no_grad():
        for step_test, (x, y_true) in enumerate(test_dataloader):
            x, y_true = x.to(device), y_true.to(device)
            if type(model) == GoogLeNetTAT:
                y_pred = model(x)[0]
            else:
                y_pred = model(x)
            # loss = loss_fun(y_pred, y_true)
            test_accuracy += (y_pred.argmax(dim=1) == y_true).sum().item()
            test_points_num += x.shape[0]

    th.manual_seed( get_random_seed() )
    
    # test_points_num  = len(test_dataloader.dataset)
    test_accuracy = test_accuracy / test_points_num * 100
    return test_accuracy


def remove_kernel(model: th.nn.Module, pair: dict[str, list, list, list],  kernel_id: int):
    target_layer_name = pair['target_layer']
    following_layers_names = pair['follow_layers']
    optional = pair['optional']
    channels_slice = [ pair['coresponding_channels_slice']['start'], pair['coresponding_channels_slice']['end'] ]
    target_layer = model.get_submodule(target_layer_name)

    # print("before pruning")
    # print(f"target {target_layer_name} -> {target_layer.weight.shape}")
    # for follow in following_layers_names:
    #    print(f"target {follow} -> {model.get_submodule(follow).weight.shape}")
    # for opt in following_layers_names:
    #    print(f"target {opt} -> {model.get_submodule(opt).weight.shape}")

    # print(f"removing :{target_layer_name} -> {following_layers_names} | {optional} | {model.get_submodule(target_layer_name).weight.shape} | {model.get_submodule(following_layers_names[0]).weight.shape}")

    with th.no_grad():
        # altering channels in the next layers
        for following_layer_name in following_layers_names:
            follow_layer = model.get_submodule(following_layer_name)
            if type(follow_layer) == th.nn.Conv2d:
                next_new_weight = th.nn.Parameter( th.cat( [follow_layer.weight[:, : channels_slice[0]+kernel_id], follow_layer.weight[:, channels_slice[0]+kernel_id+1 : ]], dim=1 ) )
                follow_layer.weight = next_new_weight
                follow_layer.in_channels = next_new_weight.shape[1]
            
            elif type(follow_layer) == th.nn.Linear:
                # defining how long is the vector created from each feature map
                featuremap_pixels_per_kernel = int(follow_layer.in_features / target_layer.out_channels)
                assert follow_layer.in_features % target_layer.out_channels == 0

                # removing i-th channel coresponds to removing "featuremap_pixels_per_kernel" kernels
                # where they start at i*"featuremap_pixels_per_kernel" position
                # and end at (i+1)*"featuremap_pixels_per_kernel" position
                # print(f"removing  kernel_id {kernel_id}\n second: {second_layer.weight.shape} -> ", end='')
                new_weight = th.nn.Parameter(th.cat([
                                                    follow_layer.weight[:, : kernel_id*featuremap_pixels_per_kernel],
                                                    follow_layer.weight[:, (kernel_id+1)*featuremap_pixels_per_kernel :]
                                                    ], dim=1 ))
                follow_layer.weight = new_weight
                follow_layer.in_features = new_weight.shape[1]
                follow_layer.out_features = new_weight.shape[0]

        # altering kernel in the target layer
        new_weight = th.nn.Parameter( th.cat( [target_layer.weight[ : kernel_id], target_layer.weight[kernel_id+1 : ]], dim=0 ) )
        target_layer.weight = new_weight
        if target_layer.bias != None:
            new_bias = th.nn.Parameter( th.cat( [target_layer.bias[ : kernel_id], target_layer.bias[kernel_id+1 : ]], dim=0 ) )
            target_layer.bias = new_bias
        target_layer.in_channels = new_weight.shape[1]
        target_layer.out_channels = new_weight.shape[0]
            

    # deleting from optional layers which are connected to target layer
    for layer_name in optional:
        layer = model.get_submodule(layer_name) 

        if type(layer) == th.nn.BatchNorm2d:
            # altering batchnormalization layer
            bn_new_weight = th.nn.Parameter( th.cat( [layer.weight[ : kernel_id], layer.weight[kernel_id+1 : ]], dim=0 ) )
            bn_new_bias = new_bias = th.nn.Parameter( th.cat( [layer.bias[ : kernel_id], layer.bias[kernel_id+1 : ]], dim=0 ) )
            bn_new_running_mean = th.cat( [layer.running_mean[ : kernel_id], layer.running_mean[kernel_id+1 : ]], dim=0 )
            bn_new_running_var = th.cat( [layer.running_var[ : kernel_id], layer.running_var[kernel_id+1 : ]], dim=0 )
            layer.weight = bn_new_weight
            layer.bias = bn_new_bias
            layer.running_mean = bn_new_running_mean
            layer.running_var = bn_new_running_var
            # information attribute update
            layer.num_features = bn_new_weight.shape[0]
        else:
            raise Exception(f"layer type ({type(layer)}) not recognized")
    
    # omitted layers are the  layers which weren't included in criterion computation
    # they are only present during resnet pruning of downsample layers channels.
    omitted_follow_layers = pair['omitted_follow_layers'] if 'omitted_follow_layers' in pair.keys() else []
    # those layers are present only for resnet for kernel removal from downsample
    omitted_parallel_layers = pair['omitted_parallel_layers'] if 'omitted_parallel_layers' in pair.keys() else []

    for omitted_layer_name in omitted_parallel_layers:
        # print(first_layer_name, second_layer_name)
        omitted_layer = model.get_submodule(omitted_layer_name)
        new_weight = th.nn.Parameter( th.cat( [omitted_layer.weight[ : kernel_id], omitted_layer.weight[kernel_id+1 : ]], dim=0 ) )
        omitted_layer.weight = new_weight
        if omitted_layer.bias != None:
            new_bias = th.nn.Parameter( th.cat( [omitted_layer.bias[ : kernel_id], omitted_layer.bias[kernel_id+1 : ]], dim=0 ) )
            omitted_layer.bias = new_bias
        omitted_layer.in_channels = new_weight.shape[1]
        omitted_layer.out_channels = new_weight.shape[0]
    
    for omitted_layer_name in omitted_follow_layers:
        omitted_layer = model.get_submodule(omitted_layer_name)
        # pruning second layer as in reverse it wouldn't be correct
        next_new_weight = th.nn.Parameter( th.cat( [omitted_layer.weight[:, : kernel_id], omitted_layer.weight[:, kernel_id+1 : ]], dim=1 ) )
        omitted_layer.weight = next_new_weight
        omitted_layer.in_channels = next_new_weight.shape[1]
        omitted_layer.out_channels = next_new_weight.shape[0]


def get_train_dataloader(batch_size: int=32, random: bool=True) -> th.utils.data.DataLoader:
    train_transform = thv.transforms.Compose([
        thv.transforms.ToTensor(),
        thv.transforms.RandomRotation(10),
        thv.transforms.RandomHorizontalFlip(0.5),
        thv.transforms.Resize([226, 226], antialias=True),
        thv.transforms.RandomCrop(size=(224, 224)),
        thv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_train_dataset = thv.datasets.CIFAR10(config.CIFAR_DATASET_PATH, train=True, transform=train_transform, download=True)
    train_dataloader = th.utils.data.DataLoader( cifar10_train_dataset, batch_size=batch_size, shuffle=random )
    return train_dataloader

def get_train_dataloader_no_aug(batch_size: int=32, random: bool=True) -> th.utils.data.DataLoader:
    train_transform = thv.transforms.Compose([
        thv.transforms.ToTensor(),
        thv.transforms.Resize([224, 224], antialias=True),
        thv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_train_dataset = thv.datasets.CIFAR10(config.CIFAR_DATASET_PATH, train=True, transform=train_transform, download=True)
    train_dataloader = th.utils.data.DataLoader( cifar10_train_dataset, batch_size=batch_size, shuffle=random )
    return train_dataloader

def get_test_dataloader(batch_size: int=128) -> th.utils.data.DataLoader:
    test_transform = thv.transforms.Compose([
        thv.transforms.ToTensor(),
        thv.transforms.Resize([224, 224], antialias=True),
        thv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_test_dataset  = thv.datasets.CIFAR10(config.CIFAR_DATASET_PATH, train=False, transform=test_transform, download=True)
    test_dataloader  = th.utils.data.DataLoader( cifar10_test_dataset, batch_size=batch_size)
    return test_dataloader


def get_test_dataloader_from_train_data(batch_size: int=128) -> th.utils.data.DataLoader:
    train_transform = thv.transforms.Compose([
        thv.transforms.ToTensor(),
        thv.transforms.Resize([224, 224], antialias=True),
        thv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_train_dataset = thv.datasets.CIFAR10(config.CIFAR_DATASET_PATH, train=True, transform=train_transform, download=True)
    sampler = th.utils.data.RandomSampler(cifar10_train_dataset, num_samples=10_000)
    dataloader = th.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=False, sampler=sampler)
    return dataloader


def choose_dataset_representatives(images_per_class: int, dataloader: th.utils.data.DataLoader, images_samples_path: str) -> None:
    '''
    This functions purpose is to choose given number of representatives for each class with
    given dataloader. The randomly choosen representatives will be then saved to given path.
    '''
    # small saving function
    def save_to_file( samples, path, name ):
        samples = list(samples.numpy())
        data = pd.DataFrame( {'image' : samples} )
        data.to_json( os.path.join(path, name), orient='records' )

    file_limit = 100
    classes = len(dataloader.dataset.classes)
    all_samples_amount = classes * images_per_class

    assert dataloader.batch_size == 1

    class_samples = []
    samples_in_file_counter = 0
    file_counter = 0
    for i in range(classes):
        samples_per_class_counter = 0
        for x, y in dataloader:
            if y == i:
                class_samples.append( x )
                samples_in_file_counter += 1
                samples_per_class_counter += 1

            if samples_in_file_counter >= file_limit:
                save_to_file( th.cat(class_samples, dim=0), images_samples_path, f"images_{file_counter:02d}" )
                file_counter += 1
                samples_in_file_counter = 0
                class_samples = []

            if samples_per_class_counter == images_per_class:
                break

    # saving what left after last file_limit crossing
    if class_samples:
        save_to_file( th.cat(class_samples, dim=0), images_samples_path, f"images_{file_counter:02d}" )

    # saving metadata
    metadata = dict()
    metadata['samples_amount'] = all_samples_amount
    pickle.dump( metadata, open(os.path.join(images_samples_path, 'metadata.pickle'), 'wb') )


def list_models(algorithm_folder_path: str, ratio: float) -> list[str]:
    ratio_folder_path = os.path.join(algorithm_folder_path, f'AlexNet_rat_{int(ratio * 100):02d}')
    model_paths = [ os.path.join(ratio_folder_path, file_name) for file_name in os.listdir(ratio_folder_path) if not file_name.endswith('.pickle') ]
    return model_paths


def print_global_pruning_ratio(pruned_model_path, references_model_path) -> None:
    model = th.load(references_model_path)
    flops_orig = calflops.calculate_flops(model=model, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=3, print_results=False)[0]
    model = th.load(pruned_model_path)
    flops_now = calflops.calculate_flops(model=model, input_shape=(1, 3, 224, 224), output_as_string=False, output_precision=3, print_results=False)[0]
    print( f"flops left: {flops_now / flops_orig * 100:5.2f} %" )


################################################################# MODELS CLASSES DEFINITIONS ##########################################################
class GlobalAveragePooling2d(th.nn.Module):
    """
    This class implements module for global average pooling
    """
    def __init__(self):
        super().__init__()

    def forward(self, input: th.Tensor):
        output = input.mean([2, 3])
        return output
    

class AlexNetGAP(th.nn.Module):
    """
    This class implements model used during AlexNet pruning
    """
    def __init__(self, class_number: int=10):
        super().__init__()
        model_alexnet = th.load(config.ALEXNET_ORIGIN_MODEL_PATH)

        self.features = copy.deepcopy(model_alexnet.features)
        self.avgpool = copy.deepcopy(model_alexnet.avgpool)
        self.classifier = th.nn.Sequential(
            th.nn.Conv2d(256, 64, kernel_size=(3, 3), padding=(1, 1), bias=True),
            th.nn.ReLU(),
            th.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            th.nn.Conv2d(64, class_number, kernel_size=(1, 1), bias=False, padding=(0, 0)),
            th.nn.Flatten()
        )
        del model_alexnet

    def forward(self, input: th.Tensor):
        x = self.features(input)
        x = self.avgpool(x)
        y = self.classifier(x)
        return y
    

class Vgg16GAP(th.nn.Module):
    def __init__(self, class_number: int=10):
        super().__init__()
        model_vgg = th.load(config.VGG16_ORIGIN_MODEL_PATH)

        self.features = copy.deepcopy(model_vgg.features)
        self.avgpool = copy.deepcopy(model_vgg.avgpool)
        self.classifier = th.nn.Sequential(
            th.nn.Conv2d(512, 64, kernel_size=(3, 3), padding=(1, 1), bias=True),
            th.nn.ReLU(),
            th.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            th.nn.Conv2d(64, class_number, kernel_size=(1, 1), padding=(0, 0), bias=False),
            th.nn.Flatten()
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        x = self.features(input)
        x = self.avgpool(x)
        output = self.classifier(x)
        return output


class ResNet50T(th.nn.Module):
    def __init__(self, class_number: int=10):
        super().__init__()
        model_resnet = th.load(config.RESNET50_ORIGIN_MODEL_PATH)

        # input
        self.conv1 = copy.deepcopy(model_resnet.conv1)
        self.bn1 = copy.deepcopy(model_resnet.bn1)
        self.relu = copy.deepcopy(model_resnet.relu)
        self.maxpool = copy.deepcopy(model_resnet.maxpool)
        # layers
        self.layer1 = copy.deepcopy(model_resnet.layer1)
        self.layer2 = copy.deepcopy(model_resnet.layer2)
        self.layer3 = copy.deepcopy(model_resnet.layer3)
        self.layer4 = copy.deepcopy(model_resnet.layer4)
        # network head
        self.avgpool = copy.deepcopy(model_resnet.avgpool)
        self.fc = th.nn.Linear(2048, class_number)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)
        return x


class DenseNet121TB(th.nn.Module):
    def __init__(self, class_number: int=10):
        super().__init__()
        model_densenet = th.load(config.DENSENET121_ORIGIN_MODEL_PATH)
        self.features = copy.deepcopy(model_densenet.features)
        self.avgpool = th.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = th.nn.Linear(in_features=1024, out_features=class_number)

    def forward(self, input: th.Tensor) -> th.Tensor:
        x = self.features(input)
        x = self.avgpool(x)
        x = th.flatten(x, 1)
        x = self.classifier(x)
        return x


class DenseNet121TAT(th.nn.Module):
    def __init__(self, class_number: int=10):
        super().__init__()
        model_densenet = th.load(config.DENSENET121_ORIGIN_MODEL_PATH)
        self.features = copy.deepcopy(model_densenet.features)
        self.classifier = th.nn.Sequential( 
            th.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            th.nn.Conv2d(in_channels=1024, out_channels=256, bias=False, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            th.nn.BatchNorm2d(256),
            th.nn.Flatten(),
            th.nn.Linear(in_features=256, out_features=class_number)
        )

    def forward(self, input: th.Tensor) -> th.Tensor:
        x = self.features(input)
        x = self.classifier(x)
        return x


class GoogLeNetTB(thv.models.GoogLeNet):
    def __init__(self, class_number: int=10):
        super().__init__(init_weights=True)
        model_googlenet = th.load(config.GOOGLENET_ORIGIN_MODEL_PATH)
        self.load_state_dict(model_googlenet.state_dict(), strict=False)

        self.aux1.fc2 = th.nn.Linear(in_features=1024, out_features=class_number)
        self.aux2.fc2 = th.nn.Linear(in_features=1024, out_features=class_number)
        self.fc = th.nn.Linear(in_features=1024, out_features=class_number)
        


class GoogLeNetTAT(thv.models.GoogLeNet):
    def __init__(self, class_number: int=10):
        super().__init__(init_weights=True)
        model_googlenet = th.load(config.GOOGLENET_ORIGIN_MODEL_PATH)
        self.load_state_dict(model_googlenet.state_dict(), strict=False)

        self.aux1.fc2 = th.nn.Linear(in_features=1024, out_features=class_number)
        self.aux2.fc2 = th.nn.Linear(in_features=1024, out_features=class_number)
        self.fc = th.nn.Sequential(
            th.nn.Conv2d(in_channels=1024, out_channels=256, bias=False, kernel_size=(1, 1), padding=(0, 0)),
            th.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            th.nn.BatchNorm2d(256),
            th.nn.Flatten(),
            th.nn.Dropout(0.2),
            th.nn.Linear(in_features=256, out_features=class_number)
        )
        del self.avgpool
        del self.dropout
    
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, Optional[th.Tensor], Optional[th.Tensor]]:
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1: Optional[th.Tensor] = None
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[th.Tensor] = None
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        try:
            x = self.fc(x)
        except Exception as e:
            print(f"!!!!!!!!!!!!!!!!! x:  {x.shape}")
            print(f"!!!!!!!!!!!!!!!!!fc: {self.get_submodule('fc.0').weight.shape}")
            raise Exception(e.__str__())
        return x, aux2, aux1


if __name__ == '__main__':
    pass
