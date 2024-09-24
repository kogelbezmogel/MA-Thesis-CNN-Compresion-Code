import os
import torch as th
import torchvision as thv
import pandas as pd
import numpy as np

import time
import datasets
import pickle
import random
import math
import sys
import copy

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls

class ThiNet():
    """ This class implements algorithm ThiNet proposed in paper: [https://arxiv.org/abs/1707.06342].

    Attributes
    ----------
    model: torch.nn.Module
        model which will be pruned
    layer_pairs: list[tuple[str]]
        pairs of layers as a list. It is used by algorithm no navigate through model
    images_samples_path: str
        path to folder with prepared samples from dataset. dataset is created with
        choose_dataset_representatives function from torchhelper package
    ratio: float
        is the ratio of how many kernels will be pruned from given layers
    train_dataloader: torch.nn.utils.DataLoader
        dataloader for train dataset
    test_dataloder: torch.nn.utils.DataLoader
        dataloder for test dataset
    samples_dataloader: torch.nn.utils.DataLoader
        dataloder for images_samples
    location_per_image: int
        it is number of samples which will be extracted from single image
    input_shapes: dict[tuple[int]]
        shapes of inputs to layers from layer_pairs list
    outputs_shapes: dict[tuple[int]]
        shapes of outputs from layers from layer_pairs list
    samples_amount: int
        amount of images in images_samples
    output_sum_mul_pairs: torch. Tensor
        it is a temporary value which is created by created_outputs method.
        It is then used for fast evaluation of subset T in find_subset method
    """

    def __init__(
            self, 
            ratio: float,
            model: th.nn.Module,
            layer_pairs: ls.LayerPairs,
            train_dataloder: th.utils.data.DataLoader,
            test_dataloder: th.utils.data.DataLoader,
            images_samples_path: str,
            locations_per_image: int,
            minimize_err: bool,
            retrain_epochs: int,
            additional_ratios_mask = None
            ) -> None:
        
        self.layer_pairs = copy.deepcopy(layer_pairs)
        self.layer_pairs_copy = copy.deepcopy(layer_pairs)
        self.images_samples_path = images_samples_path
        self.ratio = ratio
        self.minimize_err = minimize_err
        self.retrain_epochs = retrain_epochs
        self.train_dataloader = train_dataloder
        self.test_dataloader = test_dataloder
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")

        self.output_sum_mul_pairs = -1
        self.locations_per_image = locations_per_image # 10 was in the paper

        self.all_layers = set()
        for pair in self.layer_pairs:
            self.all_layers.add( pair['target_layer'] )
            for follow in pair['follow_layers']:
                self.all_layers.add(follow)

        # setting a additional ratios that can manipulate pruing ratios for layers in relate to each other
        self.additional_ratios_mask = dict()
        if additional_ratios_mask != None:
            self.additional_ratios_mask = additional_ratios_mask
        else:
            for pair in self.layer_pairs:
                t_layer_name = pair['target_layer']
                self.additional_ratios_mask[t_layer_name] = 1.0

        self.model = model.to(self.device)

        # calculating outputs for each layer
        x = th.randn([1, 3, 224, 224], dtype=th.float32).to(self.device)
        self.input_shapes = dict()
        self.output_shapes = dict()

        # hooks for collecting shapes definition
        def get_input_output_shape(name):
            def hook(model, input, output):
                self.input_shapes[name] = tuple(input[0].shape)
                self.output_shapes[name] = tuple(output.shape)
            return hook
        # adding hooks to layers
        hooks = []
        for layer_name in self.all_layers:
            hook = self.model.get_submodule(layer_name).register_forward_hook( get_input_output_shape(layer_name) )
            hooks.append(hook)
        # passing single input and removing hooks
        self.model.train()
        self.model.apply( thh.set_bn_eval )
        self.model(x)
        for hook in hooks:
            hook.remove()

        # I assume that image_set is a set where each class has its 10 examples in shape [number_of_classes * 10, image]
        # To create such a dataset function from torchhelper is used
        self.samples_amount = pickle.load(open(os.path.join(self.images_samples_path, 'metadata.pickle'), 'rb'))['samples_amount']

        # creating samples dataloder on image path
        files_list = [ os.path.join(self.images_samples_path, file_name) for file_name in os.listdir(self.images_samples_path) if file_name.startswith('images') ]
        training_paths = { 'train' : files_list }
        samples_dataset = datasets.load_dataset('json', data_files=training_paths, split='train', streaming=True)
        samples_dataset = samples_dataset.with_format('torch')
        self.samples_dataloader = th.utils.data.DataLoader(samples_dataset, batch_size=300)


    def create_inputs(self, pair: dict[str, list, list, list], seed: int=None) -> None:
        """ This function uses samples dataset to create m samples. Those are collected by propagating feature mapas till
        signal reaches input to following layer. Those feature_maps are then sampled. Amount
        of samples is determined by location_per_image argument. Each sample is the size of kernel_size in the following layer.
        Then all samples from single image are concatenated and will be treated almost like single input to the following_layer layter.
        The result of this function is tensor of shape [images_amount, channels, kernel_size, location_per_image * kernel_size]

        Arguments
        ---------
        following_layer_name: str
            ThiNet algorithm uses following layer to evaluate target layer
        seed: int
            if given samples from images are always the same
        """
        # print("\n[INFO] self.samples sizes")
        # this is the most important element in the method
        self.samples = dict()
        # self.sample finally will look accordinlgy 
        # { following_layer1_name: 
        #       "subsamples" : th.Tensor,       This is slice of channels from feature maps created by target_layer which are being inputed to the following_layer1
        #       "subkernels" : th.Tensor,       This is slice of channels of kernels from following_layer1 which is proccessing subsamples during convolution
        #   following_layer2_name:
        #       ... etc ...
        # }

        flag = True

        # if given seed random will choose always the same location from images
        if seed != None:
            random.seed(seed)

        # all following layers all the same except for number of channels (but it doesn't matter here)
        channels_slice = [pair['coresponding_channels_slice']['start'], pair['coresponding_channels_slice']['end']]
        target_layer_name = pair['target_layer']
        following_layers_names = pair['follow_layers']

        # adding hooks for gathering samples
        hooks = []
        input_handler = dict()
        # thh.set_bn_eval(self.model)
        for following_layer_name in following_layers_names:

            # creating input to sampels dictionary
            self.samples[following_layer_name] = dict()

            # extracting needed layer
            following_layer = self.model.get_submodule( following_layer_name )

            # creating list of spatial locations for images
            following_layer_input_shape = self.input_shapes[following_layer_name][ -2: ]

            # asserts that all shapes like padding, kernel_size have equal widht and height
            assert following_layer_input_shape[0] == following_layer_input_shape[1]
            assert following_layer.padding[0] == following_layer.padding[1]
            assert following_layer.kernel_size[-2] == following_layer.kernel_size[-1]
            assert following_layer.stride[0] == following_layer.stride[1]
            assert following_layer.kernel_size[-1] % 2 == 1 # to make sure that kernel size is not even

            # kernel_border_width = (following_layer.kernel_size[-1] - 1) // 2
            kernel_width = following_layer.kernel_size[-1]
            # kernel_depth = following_layer.weight.shape[1]

            # registering hook for gathering inputs for following layer
            def get_input_to_layer_hook(name: str, channels_slice_start: int, channels_slice_end: int):
                def hook(model, input):
                    # maps created by scored kernels are created only within given slice
                    # print(f"{name} input max: {input[0].max()}")
                    input_handler[name] = input[0][:, channels_slice_start : channels_slice_end, :, :] # slicing channels produced by target layer
                return hook
            hook_handler = self.model.get_submodule(following_layer_name).register_forward_pre_hook( get_input_to_layer_hook(following_layer_name, channels_slice[0], channels_slice[1]) )
            hooks.append(hook_handler)

            # allocating memory for collecting samples (locations)
            s_size = [self.samples_amount * self.locations_per_image, channels_slice[1]-channels_slice[0], kernel_width, kernel_width]
            s_size = tuple( int(v) for v in s_size )
            self.samples[following_layer_name]["subsamples"] = th.empty(s_size, dtype=th.float32) #[images_num * locations_perimage, channels, kernel_height, kernel_width]
            self.samples[following_layer_name]["subkernels"] = following_layer.weight[:, channels_slice[0] : channels_slice[1], :, :].detach().clone()

        # gathering samples for each following layer separately
        for step, batch in enumerate(self.samples_dataloader):
            start = step * self.samples_dataloader.batch_size
            end =  min( (step + 1) * self.samples_dataloader.batch_size, self.samples_amount )

            # propageting signal till following_layer
            self.model = self.model.to(self.device)
            self.model.train()
            with th.no_grad():
                x = batch['image'].to(self.device)
                self.model(x)
            
            for f_layer_name in following_layers_names:
                # print( f"{f_layer_name}")
                # accessing slice of feature_maps
                input = input_handler[f_layer_name]
                # padding the input which is needed for sampling 
                following_layer = self.model.get_submodule( f_layer_name )           
                transform = thv.transforms.Pad(following_layer.padding)
                x = transform(input)

                # definig sampling points for this f_layer and this batch 
                stride_width = following_layer.stride[1]
                kernel_border_width = (following_layer.kernel_size[-1] - 1) // 2
                input_width = self.input_shapes[f_layer_name][-1]
                padding_width = following_layer.padding[1]
                accessible_locations_number = math.ceil((input_width + 2*padding_width - 2*kernel_border_width) / stride_width)
                accessible_sampling_start = int(kernel_border_width)
                possible_points = range(accessible_locations_number * accessible_locations_number)

                for image_id in range( x.shape[0] ):
                    # for each image 10 points are selected randomly
                    try:
                        points = random.sample( possible_points, self.locations_per_image )
                    except:
                        print(f"{f_layer_name} | pos points {len(possible_points)}  ") # fc.0 input is to small
                        points = [p for p in possible_points]
                    points = [ np.array(divmod(point, accessible_locations_number)) * stride_width + accessible_sampling_start for point in points ]

                    image = x[image_id]
                    for point_id, point in enumerate(points):
                        width = ( int(point[0] - kernel_border_width), int(point[0] + kernel_border_width + 1) )
                        height = ( int(point[1] - kernel_border_width), int(point[1] + kernel_border_width + 1) )
                        extracted_sample = image[:, width[0] : width[1], height[0] : height[1]] # [channels, width, height]
                        # here as a last step for each image his 10 locations are being gathered to tensors [channels, kernel_size, 10 * kernel_size]
                        # therefore, for processing single sample convolution with stride=kernel_width and groups=channels can be used.
                        self.samples[f_layer_name]["subsamples"][self.locations_per_image * (start + image_id) + point_id, :, :, : ] = extracted_sample.detach().cpu()

            # printing created samples metadata
            # for f_layer_name in following_layers_names:
            #     print( f"{f_layer_name:40s} {self.samples[f_layer_name]['subsamples'].element_size() * self.samples[f_layer_name]['subsamples'].nelement() / 1_000_000:8.1f} MB | {self.samples[f_layer_name]['subsamples'].shape}")
            #     print( f"{'subkernels':40s} {self.samples[f_layer_name]['subkernels'].element_size() * self.samples[f_layer_name]['subkernels'].nelement() / 1_000_000:8.1f} MB | {self.samples[f_layer_name]['subkernels'].shape}")
            # print()
            for hook_handler in hooks:
                hook_handler.remove()
            # thh.set_bn_train(self.model)
            # thh.reset_bn_stats(self.model)


    def create_outputs(self, pair: dict[str, list, list, list]) -> None:
        """This functions creates different type of output than previous one
        In the prevoious attempt outputs are intermediate product during convolution which are during finding subset summed along subset of kernel channels than squered and again summed along all samples.
        That means for each subset of channels algorithm needs to iterate through whole dataset each time. During finding subset amount of subsets to check is a combination of k-kernels to prune from c-kernels in the target layer.
        In this attempt to speed up the whole process of iterating through dataset over and over is omitted. In the addition most of the opertaions of summations and exponetations is done ahead.

        The whole data is stored in a list of size: c + c * (c-1) where c is amount of kerenls in the target layer (channels in following layer)
        This include 'c' of summed squers of x_i for all m samples and 'c(c-1)' of summmed 2(x_i)(x_j) whrere i != j and

        Arguments
        ---------
        following_layer_name: str
            ThiNet algorithm uses following layer to evaluate target layer
        """
        print(f"[INFO] creating outputs")
        # empty matrix for final kernel scores 
        target_layer_name = pair['target_layer']
        following_layers_names = pair['follow_layers']
        
        channels = self.model.get_submodule(target_layer_name).weight.shape[0]
        self.output_sum_mul_pairs = th.zeros( [channels, channels], dtype=th.float64) # for 2*x_i*x_j

        # calculating scaling factor for output_sum_mul_pairs
        scaling_factor = 0
        for f_layer_name in following_layers_names:
            scaling_factor += self.samples[f_layer_name]['subsamples'].shape[0]

        for f_layer_num, f_layer_name in enumerate(following_layers_names):
            # gathering informations
            following_layer_subweight = self.samples[f_layer_name]['subkernels']
            kernels = following_layer_subweight.shape[0]
            # channels = following_layer_subweight.shape[1]

            # This changed dimensionality will be necessery in manually conducted convolution to use groups param.
            following_layer_subweights_shifted = th.movedim(following_layer_subweight, 0, 1)

            # usage of meshgrid doesn't allow to procces the data in one step and it must be slices to batches
            dataset = th.utils.data.TensorDataset(self.samples[f_layer_name]['subsamples'])
            batch_size = 256
            dataloader = th.utils.data.DataLoader(dataset, batch_size=batch_size)

            with th.no_grad():
                # processing data batch by batch
                for step, batch in enumerate(dataloader):
                    batch = batch[0].to(self.device)

                    # to be able to use groups param in conv functions each kernel must be taken separately and be in proper shape [channels, 1, width, height]
                    for kernel_id in range(kernels):
                        zombie_kernel_weights = following_layer_subweights_shifted[:, kernel_id:kernel_id+1].to(self.device)
                        output = th.nn.functional.conv2d(batch, weight=zombie_kernel_weights, bias=None, padding=(0, 0), groups=channels, stride=(1, 1)) # why padding was (1, 1)????????????????
                        output = th.movedim(output, 1, -1)                                      # [batch, 1, 1, channels]
                        output = th.reshape(output, [-1, channels])                             # [samples, channels]

                        # This step need to be explained
                        # What I am trying to achive here is for every row in batch to create specific matrix.
                        # This matrix will contain all posibble pairs of multiplications for elements in row.
                        # Each element in created matrix will be then (x_i * x_j) where x_i is i-th element in row and x_j is j-th element in the same row.
                        # This way creating squared sum of subset of elements will be faster to calculate.
                        # For example for row [1, 2, 3, 4] I want (1, 2, 4)^2. To do so I only need to zero out 3-th row and 3-th column of matrix and sum all elements that are left there.
                        row_indices = th.arange(channels)
                        meshgrid_a, meshgrid_b = th.meshgrid([row_indices, row_indices], indexing='ij')
                        rows_1 = output[:, meshgrid_a]
                        rows_2 = output[:, meshgrid_b]
                        # rows_1, rows_2 are a sets of matrices. Each matrix coresponds to one row in batch.
                        # in rows_1 single matrix has coresponding row cloned and stacked on each other creating c-rows
                        # in rows_2 single matrix has coresponding row cloned and stacked next to each other creating c-columns

                        self.output_sum_mul_pairs += (rows_1 * rows_2).sum(dim=0).detach().cpu().clone() / scaling_factor # multipling creates batch of matrices where each matrix has all combinations x_i*x_j for coresponding row
                        # summing along dim=0 reduces the batch to single a matrix. where (i, j) element is the sum of (x_i * x_j) for all samples in the dataset.

                        if f_layer_num==0 and kernel_id==0 and step==0:
                            # print( f" 1 of {kernels} kernels took time:{end-start}s")
                            # print( f"batch shape: {batch.shape}")
                            # print( f"single meshgrid shape: {rows_1.shape}")
                            print( f"both meshgrid_size: {2 * rows_1.nelement() * rows_1.element_size() / 1_000_000:8.1f} MB")


    def find_subset_T(self, pair: dict[str, list, list, list], number: int=None) -> tuple:
        """ It is direct implementation of algorithm from paper. It goes through all
        possible subsets of T (I is the opposite subset) and scores them to select T which
        minimize error created by lack of T kernels in target channel.

        Arguments
        ---------
        following_layer_name: str
            name of the layer succeding targeted layer
        number: int
            it is number of pruned kernels in target layer. If it is None
            then number is defined by ratio and number of kernels in layer
        """
        target_layer_name = pair['target_layer']
        
        num_channels = self.model.get_submodule(target_layer_name).weight.shape[0]
        I, T = [ i for i in range(num_channels) ], []

        # number of kernels to remove can be explicitly passed as a argument during sensitivity analysis
        kernels_to_remove_num = -1
        if number != None:
            kernels_to_remove_num = number
        elif number == None:
            print("\n[INFO] finding subset to prune")
            kernels_to_remove_num = int(round(num_channels * self.ratio * self.additional_ratios_mask[target_layer_name], 0))
            print(f"number of kernels to prune: {self.ratio * self.additional_ratios_mask[target_layer_name]} => kernels {kernels_to_remove_num}")

        
        while len(T) < kernels_to_remove_num:
            min_value = float('inf')
            min_i = -1

            for num, channel in enumerate(I):
                I.remove(channel)
                value = self.channels_score( I )
                I.append(channel)

                if value < min_value:
                    min_value = value
                    min_i = channel

            T = T + [min_i]
            I.remove(min_i)
        return T, I


    def channels_score(self, I: list) -> th.Tensor:
        ''' This function scores channels accorindg to (6) equation from paper.

        Arguments
        ---------
        I: list[int]
            this is the list of channels which are not taken into cosideratrion for pruning.
            It is the oposite of T. sum(I, T) = all kernels in target layer
        '''
        value = 0
        with th.no_grad():
            mask = th.ones_like(self.output_sum_mul_pairs, dtype=th.bool)
            mask[:, I] = False
            mask[I, :] = False
            value = (self.output_sum_mul_pairs * mask).sum()
        return value


    def prune_target_layer(self, T: list[int], pair: dict[str, list, list, list]) -> None:
        '''Functions removes kernels in the target layer and channels in the following layer.

        Arguments
        ---------
        T: list[int]
            kernels choosen to br pruned from target layer
        target_layer_name: str
            layer from which kernels will be remove
        following_layer_name: str
            layer_succeding target layer. It will have channels remove to match new shape of signal
        batchn_layer_name
            batch normalization layer lying after target layer befor following layer.
            If None that means there is none there
        '''
        T.sort(reverse=True) # cutting the last kernels first keeps the right order of the rest kernels
        for kernel_id in T:
            thh.remove_kernel(self.model, pair, kernel_id)
            ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)

    
    def prune_and_minimize_error(self, T: list[int], pair: dict[str, list, list, list]):
        """ This function can be used after pruning each layer to further minimize occured error.
        To achieve this functions is using the ordinary least squares approach to find scaling factor for channels from kernels
        It is clearer descripted in paper in equation (7). As a additional result function also prunes the target layer

        Arguments
        ---------
        T: list[int]
            kernels choosen to be pruned from target layer
        target_layer_name: str
            layer from which kernels will be remove
        following_layer_name: str
            layer_succeding target layer. It will have channels remove to match new shape of signal
        batchn_layer_name
            batch normalization layer lying after target layer befor following layer.
            If None that means there is none there
        """
        self.model = self.model.to(self.device)
        target_layer_name = pair['target_layer']
        following_layers_names = pair['follow_layers']
        
        # pruning layer
        self.prune_target_layer(T, pair)

        print(f"Minimalization for {target_layer_name}")
        y_ref_unpruned_all = []
        X_all = []
        # traintest = thh.get_test_dataloader_from_train_data()
        # print(f"wo mini test_acc: {thh.evaluate_model(self.model, self.test_dataloader)}")
        # print(f"wo mini train_acc: {thh.evaluate_model(self.model, traintest)}")
        
        for following_layer_name in following_layers_names:
            zombie_weight = self.samples[following_layer_name]['subkernels']
            zombie_following_layer = th.nn.Conv2d(in_channels=zombie_weight.shape[1], out_channels=zombie_weight.shape[0], kernel_size=zombie_weight.shape[-1])
            zombie_following_layer.weight = th.nn.Parameter(zombie_weight)
            # gathering y_i from unprunde layer
            zombie_following_layer.stride = (1, 1)
            zombie_following_layer.padding = (0, 0)
            zombie_following_layer.bias = None
            y_ref_unpruned = zombie_following_layer(self.samples[following_layer_name]['subsamples'].to(self.device))
            y_ref_unpruned_all.append( th.reshape(y_ref_unpruned.double(), [-1]) )

            # pruning coresponding channels from samples
            T.sort(reverse=True)
            # this will be used to restore samples after algorithm finishes
            deleted_samples_columns = []
            for channel_id in T:
                deleted_samples_columns.insert( 0, (channel_id, self.samples[following_layer_name]['subsamples'][:, channel_id:channel_id+1, :, :]) )
                self.samples[following_layer_name]['subsamples'] = th.cat( 
                    [self.samples[following_layer_name]['subsamples'][:, :channel_id, :, :],
                     self.samples[following_layer_name]['subsamples'][:, channel_id+1:, : ,:]]
                     , dim=1
                     )
                # removing channels from zombie following layer        
                zombie_following_layer.weight = th.nn.Parameter( th.cat([zombie_following_layer.weight[:, : channel_id], zombie_following_layer.weight[:, channel_id+1 : ]], dim=1) )

            # to check if minimalization really takes place
            # y_wo_mini = zombie_following_layer(self.samples[following_layer_name]['subsamples'].to(self.device))


            with th.no_grad():
                # y_ref = th.reshape(y_ref, [-1, 1]).double()
                m = self.samples[following_layer_name]['subsamples'].shape[0]
                n, c, w, h = zombie_following_layer.weight.shape
                weights = zombie_following_layer.weight.detach().clone()
                # moving dimension because in otherwise kernel channels won't match coresponding input channels in grouped convolution
                weights = th.movedim(weights, 0, 1)
                weights = th.reshape(weights, [n*c, 1, w, h])
                # print(f"err: {th.pow(weights[5, 0] - layer.weight[1, 1], 2).sum()}")
                X = th.conv2d(self.samples[following_layer_name]['subsamples'].to(self.device), weights, padding=(0, 0), groups=c).detach() 
                # columns in X are in wrong order. they are grouped channels wise but should be kernel wise
                X = th.reshape(X, [m, n*c])
                # creting proper ordering of columns. From c chanels by n kernels to n kernels by c channels
                order = [ i+j*n for i in range(n) for j in range(c)] 
                X = X[:, order]
                X_all.append( th.reshape(X, [m * n, c]).double() )
                
                # restoring the samples tensor
                for (channel_id, d_columns) in deleted_samples_columns:
                    self.samples[following_layer_name]['subsamples'] = th.cat( 
                        [self.samples[following_layer_name]['subsamples'][:, :channel_id, :, :],
                         d_columns, self.samples[following_layer_name]['subsamples'][:, channel_id:, : ,:]],
                         dim=1
                         )
                deleted_samples_columns.clear()


        with th.no_grad():
            # usage of double is dictated by enormous values in inverted matrix.
            # the bigger is the disproportion between target_layer in_channels and follow_layer in_channels the bigger are the values.
            X = th.cat(X_all, dim=0)
            print(f"[INFO] minimizing X: {X.shape} | {X.dtype} | {X.element_size() * X.nelement() / 1e9:.3f} GB")
            y_ref_unpruned = th.cat(y_ref_unpruned_all, dim=0)  

            X_t = th.transpose(X, dim0=0, dim1=1)
            XX = th.linalg.matmul( X_t, X )
        
            if th.linalg.det(XX) == 0:
                print("[INFO] adding noise")
                X = X + th.randn(X.shape, dtype=th.float64).to(self.device) * 1e-10 # adding noise of magnitude 1e-10
                X_t = th.transpose(X, dim0=0, dim1=1)
                XX = th.linalg.matmul(X_t, X)
            
            target_layer = self.model.get_submodule(target_layer_name)
            try:
                w = th.linalg.matmul(
                                    th.linalg.inv(XX),
                                    th.linalg.matmul( X_t, y_ref_unpruned )
                                    )
            except:
                print("[WARN] not minimized (still Det(XX) == 0 after noise)")
                w = th.ones([target_layer.weight.shape[0]])
            
            # XX_inv = th.linalg.inv(XX)
            # print("[INFO] mini stats:")
            # print(f"XX mean : {XX.mean():8.5f} | {XX.min():8.5f} | {XX.max():8.5f}")
            # print(f"XX inv mean : {XX_inv.mean():8.5f} | {XX_inv.min():8.5f} | {XX_inv.max():8.5f}")
            # print(f"w mean      : {w.mean()} | {w.min()} | {w.max()} ")
            
            if th.abs(w.min()) < 1e2 and w.max() < 1e2:
                for j in range(len(w)):
                    target_layer.weight[j : j+1, :, :, :] = w[j] * target_layer.weight[j : j+1, :, :, :]
                    # following_layer.weight[:, channels_slice[0]+j, :, :] = w[j] * following_layer.weight[:, channels_slice[0]+j, :, :]
            else:
                print("[WARN] not minimized (Too big values in coefficients)")    
            # print(f"w mini test_acc: {thh.evaluate_model(self.model, self.test_dataloader)}")
            # print(f"w mini train_acc: {thh.evaluate_model(self.model, traintest)}")


    def sensitivity_analysis(self, portion_size: float=0.1) -> dict[list]:
        """ This functions conducts sensitivity analysis. It uses ThineNet algorithm to prune single layer
        furhter and furtner each time saving evaluation how pruned layer affected accuracy of the model.
        After that model is restored and another layer is being analyzed.

        Arguments
        ---------
        portion_size: float
            is the size ratio for each layer of how much it will be further pruned.
            Default is 0.1 whant means for single layer analysis 1.0, 0.9, 0.8, ... kernels left
        """
        # results will be collected for each layer as a list of succesive reductions by portion_size of kernels.
        self.sensitivity_results = dict()
        model_starting_point = copy.deepcopy(self.model)
        # preparing reduction points
        reductions = [val / 100 for val in range(int(portion_size * 100), 100, int(portion_size * 100))] #!!!!!!!!!!!!!!!!!
        # starting eveluation
        start_acc = thh.evaluate_model(self.model, self.train_dataloader)

        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            print(f"{t_layer_name} analysis -------------------------------")
            layer_start = time.time()
                
            all_kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]
            self.sensitivity_results[t_layer_name] = [ (0.0, start_acc) ]

            self.create_inputs(pair)
            self.create_outputs(pair)

            for reduction in reductions:
                # loading model starting point is necessary as after pruning and retraining created outputs do not match.
                kernel_portion_size = round(all_kernels_num*reduction, 0)
                # defining the best subset for pruning
                T, I = self.find_subset_T(pair, number=kernel_portion_size)
                # pruning found subset and minimizing error
                if self.minimize_err:
                    self.prune_and_minimize_error(T, pair)
                else:
                    self.prune_target_layer(T, pair)
                # retraining model
                thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs, return_best_test=False, return_best_train=True)
                # evaluating model
                eval_acc = thh.evaluate_model(self.model, self.train_dataloader)
                # saving results
                self.sensitivity_results[t_layer_name].append( ( round(reduction, 2), eval_acc) )
                # restarting model and layer_pairs
                self.model = copy.deepcopy(model_starting_point)
                self.layer_pairs = copy.deepcopy(self.layer_pairs_copy)
                print(f"done layer: {t_layer_name} | red: {reduction} | {pair['coresponding_channels_slice']}")
            
            layer_end = time.time()
            print(f"{t_layer_name} analysis: {round(layer_end-layer_start, 1)}s ---------------------------done\n")
        return self.sensitivity_results


    def prune_model(self) -> None:
        """ This function utilises defined functions and conducts pruning layer by layer

        Arguments
        ---------
        minimize_error: bool
            defines if the algorithm uses additional error minimalization mechanism
        """
        # iterating over layers_pairs for pruning
        self.model.apply( thh.set_bn_eval )
        for pair in self.layer_pairs:

            start_all = time.time()

            print( f"layer ( {pair['target_layer']} ) pruning starts ----------")

            # creating inputs for follwing_layer
            self.create_inputs(pair)
            # creating outputs and transforming them into sum of multiplication pairs
            self.create_outputs(pair)
            # finding subset
            try:
                T, I = self.find_subset_T(pair)
            except Exception as e:
                print(e)
                print(f"{self.output_sum_mul_pairs}")
                raise Exception("?")

            # pruning and minimizing error
            if self.minimize_err:
                self.prune_and_minimize_error(T, pair)
            else:
                self.prune_target_layer(T, pair)

            # retraining model
            self.model.apply( thh.set_bn_train )
            thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs)
            self.model.apply( thh.set_bn_eval )
            end_all = time.time()
            print( f"layer ( {pair['target_layer']} ) pruned and tuned in {end_all - start_all:5.0f}s ----------")