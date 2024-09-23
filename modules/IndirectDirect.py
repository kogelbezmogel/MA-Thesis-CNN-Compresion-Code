import os
import torch as th
import time

import sys
import copy
import math

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls

class IndirectDirect():
    """ This class implements algoritm proposed in papaer [https://www.sciencedirect.com/science/article/abs/pii/S092523122301247X]

    Attributes
    ----------
    model: torch.nn.Module
        model which will be pruned
    layers_pairs: list[tuple[str]]
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
    pruning_list: dict[list[int]]
        temporary variable which is a dictionary of lists of kernels to prune in each layer
    """

    def __init__(
                    self,
                    ratio: float,
                    model: th.nn.Module,
                    layer_pairs: ls.LayerPairs,
                    train_dataloder: th.utils.data.DataLoader,
                    test_dataloder: th.utils.data.DataLoader,
                    retrain_epochs: int,
                    max_reduction_ratio: float = 1.0
                ):
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_dataloader = train_dataloder
        self.test_dataloader = test_dataloder
        self.layer_pairs = copy.deepcopy(layer_pairs)
        self.layer_pairs_copy = copy.deepcopy(layer_pairs)
        self.pruning_list = None
        self.ratio = ratio
        self.retrain_epochs=retrain_epochs
        self.max_reduction_ratio = max_reduction_ratio


    def __calculate_norms_for_kernels_set(self, weights: th.Tensor) -> th.Tensor:
        """ This function returns tensor of shape [kernels_num, kernel_num]. This tensor contains norms for
        kernels in layer. In each (i, j) place in tensor is L2_norm( w_i, w_j ) where w_i and w_j are kernel i-th and j-th

        Arguments
        ---------
        weights: torch.Tensor
            this is set of torch tensor weights of shape [kernels_amount, channels, width, height] from layer
        """
        with th.no_grad():
            kernel_amount = weights.shape[0]
            indices = th.arange(kernel_amount)
            rows, cols = th.meshgrid([indices, indices], indexing='ij')

            kernels_row_wise = weights[rows]
            kernels_col_wise = weights[cols]

            norms = th.subtract( kernels_row_wise, kernels_col_wise )
            norms = th.pow(norms, 2)
            norms = th.sum(norms, dim=[2, 3, 4])
            norms = th.sqrt(norms) # now in the result matrix value on (i, j) place should be L2_norm( w_i, w_j ). Only one traingle slice is needed then.

            triangle_rows, traingle_cols = th.triu_indices( kernel_amount, kernel_amount)
            norms[triangle_rows, traingle_cols] = 0 # diagonal is always 0 so zeroing upper traingle.
        return norms


    def __get_kernels_gvalues(self, norms_l2: th.Tensor) -> th.Tensor:
        """ This functions calculates g_values according to (1) equation from paper

        Arguments
        ---------
        norms_l2: torch.Tensor
            norms calculated by __calculate_norms_for_kernels_set function
        """
        kernels_amount = norms_l2.shape[0]
        gvalues = th.empty([kernels_amount], dtype=th.float64)

        for j in range(kernels_amount):
            mask = th.zeros([kernels_amount, kernels_amount], dtype=th.int8).to(self.device)
            mask[j, :] = 1
            mask[:, j] = 1

            gvalues[j] = th.sum(mask * norms_l2)
        return gvalues


    def __find_subset_to_prune(self, pair: dict, amount_to_prune: int) -> th.Tensor:
        """ This function calculates direct and indirect effect by utilising __calculate_norms_for_kernels_set
        and then __get_kernels_gvalues on extracted kernels sets. As a result it returns for each layer list
        of kernels to prune

        Arguments
        ---------
        target_layer_name: str
            layer from which kernels will be remove
        following_layer_name: str
            layer_succeding target layer. It will have channels remove to match new shape of signal
        """

        target_layer_name = pair['target_layer']
        follow_layers_names = pair['follow_layers']
        channels_slice = [pair['coresponding_channels_slice']['start'], pair['coresponding_channels_slice']['end']]
        target_weights = self.model.get_submodule(target_layer_name).weight.detach().clone()

        kernel_num = target_weights.shape[0]

        zombie_kernels = []
        # gathering slices from all following layers if more than 1. If not the slice if the whole weight from single follow layer
        for follow_layer_name in follow_layers_names:
            follow_weights_slice = self.model.get_submodule(follow_layer_name).weight[:, channels_slice[0] : channels_slice[1], :, :].detach().clone()
            zombie_kernels.append( follow_weights_slice )

        zombie_kernels = th.cat(zombie_kernels, dim=0)
        # creating artificial kernels by stacking j-th channel from every kernel on each other
        zombie_kernels = th.movedim(zombie_kernels, 0, 1)

        # direct effects
        direct = th.empty([kernel_num], dtype=th.float64)
        direct_norms = self.__calculate_norms_for_kernels_set(target_weights)
        direct_gvalues = self.__get_kernels_gvalues(direct_norms)

        g_min = direct_gvalues.min()
        g_max = direct_gvalues.max()
        direct = th.divide(direct_gvalues-g_min, g_max-g_min)

        # indirect effects
        indirect = th.empty([kernel_num], dtype=th.float64)
        indirect_norms = self.__calculate_norms_for_kernels_set(zombie_kernels)
        indirect_gvalues = self.__get_kernels_gvalues(indirect_norms)

        g_min = indirect_gvalues.min()
        g_max = indirect_gvalues.max()
        indirect = th.divide(indirect_gvalues-g_min, g_max-g_min)

        # set of hyperparameters for evaluation cryteria (defined in paper)
        alpha = 0.1
        gamma = 1.0

        final_scores = alpha*direct + (1-alpha)*indirect - th.pow( th.abs(direct-indirect), gamma )

        # picking the subset to prune
        kernels_to_prune = th.topk(final_scores, amount_to_prune, largest=False).indices
        return kernels_to_prune


    def __remove_kernels_channels(self) -> None:
        """ This functions prunes model on base of pruning_list attribute

        Arguments
        ---------
        target_layer_name: str
            layer from which kernels will be remove
        following_layer_name: str
            layer_succeding target layer. It will have channels remove to match new shape of signal
        batchn_layer_name
            batch normalization layer lying after target layer befor following layer.
            If None that means there is none there
        """
        for pair in self.layer_pairs:
            target_layer_name = pair['target_layer']
            # finding pruning list for tartget layer
            if not self.pruning_list:
                raise SystemExit("Pruning list is empty")
            T = self.pruning_list[target_layer_name]
            # sorting kernels to keep their correct order during removal
            T.sort(reverse=True)

            # removing kernels one by one starting from the farest ones in order.
            for kernel_id in T:
                thh.remove_kernel(self.model, pair, kernel_id)
                ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)


    def sensitivity_analysis(self, portion_size: float=0.1) -> dict[list]:
        """ This functions performs sensitivity analysis for each layer separately.
        Results are save as a class atribute and also returned.

        Argumetns
        ---------
        portion_size: float
            means the step size during pruning for single layer
        """
        # results will be collected for each layer as a list of succesive reductions by portion_size of kernels.
        self.sensitivity_results = dict()
        model_starting_point = copy.deepcopy(self.model)
        ratio_copy = self.ratio
        # preparing reduction points
        reductions = [val / 100 for val in range(int(portion_size * 100), 100, int(portion_size * 100))]
        # starting eveluation
        start_acc = thh.evaluate_model(self.model, self.test_dataloader)

        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            print(f"{t_layer_name} analysis -------------------------------")
            layer_start = time.time()
            self.sensitivity_results[t_layer_name] = [ (0.0, start_acc) ]

            for reduction in reductions:
                self.ratio = reduction
                reduction_plan = self.create_reduction_plan()
                print(f"layer: {t_layer_name} | red: {reduction} | red_plan: {reduction_plan[t_layer_name]}")

                for step in range(reduction_plan['steps']):
                    amount_to_prune = reduction_plan[t_layer_name][step]
                    print("sending pair ", pair)
                    T = self.__find_subset_to_prune(pair, amount_to_prune)

                    self.pruning_list = dict()
                    # setting lists for other layers to be empty during kernels removal
                    for row in self.layer_pairs:
                        layer_name = row['target_layer']
                        self.pruning_list[layer_name] = []
                    # setting only target layer layer to have full list
                    self.pruning_list[t_layer_name] = list(T)

                    # removing kernels according to created pruning list
                    self.__remove_kernels_channels()
                    if step < reduction_plan['steps']-1:
                        thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=1, return_best_test=False, return_best_train=True)
                # model retraining and evaluation
                thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs, return_best_test=False, return_best_train=True)
                test_acc = thh.evaluate_model(self.model, self.test_dataloader)
                self.sensitivity_results[t_layer_name].append( (round(reduction, 2), test_acc) )

                # restarting model
                self.model = copy.deepcopy(model_starting_point)
                # restraing layer_pairs with current pair
                self.layer_pairs = copy.deepcopy(self.layer_pairs_copy)
                pair = self.layer_pairs[t_layer_name]
            layer_end = time.time()
            print(f"{t_layer_name} analysis: {round(layer_end-layer_start, 1)}s ---------------------------done\n")
    
        self.ratio = ratio_copy
        return self.sensitivity_results

    
    def create_reduction_plan(self):
        reduction_plan = dict()
        steps = math.ceil(self.ratio / self.max_reduction_ratio)
        reduction_plan['steps'] = steps

        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]
 
            # second algorithm attempt
            max_kernels_portion = math.ceil(self.max_reduction_ratio * kernels_num)
            kernels_to_remove = int(round(self.ratio * kernels_num, 0))
            layer_reduction_list = [ min(kernels_to_remove-i*max_kernels_portion, max_kernels_portion) for i in range(steps) ]
            layer_reduction_list = [portion if portion >= 0 else 0 for portion in layer_reduction_list]
            reduction_plan[t_layer_name] = layer_reduction_list
        
        return reduction_plan


    def prune_model(self) -> None:
        """ This functions performs while pruning according to layers_pairs given
        by given ratio
        """
        reduction_plan = self.create_reduction_plan()
        print(f"reduction plan:")
        for key in reduction_plan.keys():
            print(f"{key:30s} | {reduction_plan[key]}")
        print("-----------------")

        # finding kernels sets which will be pruned
        self.pruning_list = dict()
        for step in range(reduction_plan['steps']):
            print(f"step {step}")
            for pair in self.layer_pairs:
                target_layer_name = pair['target_layer']
                amount_to_prune = reduction_plan[target_layer_name][step]
                if amount_to_prune > 0:
                    T = list( self.__find_subset_to_prune(pair, amount_to_prune) )
                else:
                    T = []
                self.pruning_list[target_layer_name] = T
            
            self.__remove_kernels_channels()
            # retraining after each step except the last one
            if step < reduction_plan['steps']-1:
                thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs)

        # removing kernels
        self.model = self.model.cpu()
        print( f"model pruned")
