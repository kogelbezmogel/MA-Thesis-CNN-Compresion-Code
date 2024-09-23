import torch as th
import numpy as np

import copy
import sys
import calflops

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls

class Random:
    """ This class implements random pruning algorithm. It is assures that there is at least 10 kernels in left.
    Random selection of kernels is weighted by the amount of parameters in layer to total sum of them in model.

    Attributes
    ----------
    model: torch.nn.Module
        model given to the class to be pruned
    goal_flops_ratio: float
        ratio of flops from original model to pruned model which algorithm i aimming to achieve
    layers_pairs: list[tuple[str]]
        pairs of layers as a list. It is used by algorithm no navigate through model
    input_shape: tuple[int]
        shape of images that are being used as input to the model
    """

    def __init__(self, goal_flops_ratio: float, model: th.nn.Module, layers_pairs: ls.LayerPairs):
        self.model = model
        self.goal_flops_ratio = goal_flops_ratio
        self.layers_pairs = copy.deepcopy(layers_pairs)
        self.input_shape = (1, 3, 224, 224)
        self.flops_origin = calflops.calculate_flops(self.model, self.input_shape, print_results=False, output_as_string=False)[0]


    def select_layer(self, x: float, cumulative_probabilities: th.Tensor) -> tuple[tuple[str], int]:
        """ This function returns weighted randomly selected layer.

        Arguments
        ----------
        x: float
            random sample from randge[0, 1] which will be projected on layer
        cumulative_probabilities: th.Tensor
            as probability here is number of paramters in layer by all paramters in all layers.
            This argument is cumulation of this probability distribution.
        """
        for i, prob in enumerate(cumulative_probabilities):
            if x <= prob:
                return (self.layers_pairs.get_by_id(i), i)
        print(x, cumulative_probabilities)
        raise Exception("No x choosen")


    def prune_model(self) -> None:
        """
        This functions conducts full pruning based on parameters given in constructor.
        It results with modification of given model.
        """
        current_params_list = dict()
        all_params = 0
        all_kernels = 0

        for pair in self.layers_pairs:
            t_layer_name = pair['target_layer']
            
            weight_size = th.tensor( self.model.get_submodule(t_layer_name).weight.shape, dtype=th.int32 )
            kernel_amount = weight_size[0]
            all_kernels += kernel_amount

            all_params_in_layer = th.prod( weight_size, dtype=th.int64 )
            params_per_kernel = th.prod( weight_size[-3:], dtype=th.int64 )
            bias = (self.model.get_submodule(t_layer_name).bias != None)

            all_params_in_layer = th.prod( weight_size, dtype=th.int64 )
            all_params += all_params_in_layer
            current_params_list[t_layer_name] = {'params_amount' : int(all_params_in_layer), 'kernel_amount' : int(kernel_amount), 'params_per_kernel' : int(params_per_kernel), 'bias' : bias}

        flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
        current_flops_ratio = flops_now / self.flops_origin

        # pruning loop
        kernels_counter = 0
        flops_update_timer = 20
        while current_flops_ratio > self.goal_flops_ratio:
            # creating probabilities for weighting the random selection of layer
            probabilities = th.tensor([ current_params_list[pair['target_layer']]['params_amount'] / all_params for pair in self.layers_pairs ])
            cumulative_probabilities = th.cumsum(probabilities, dim=0)

            # choosing the layer
            while True:
                layer_rand = th.rand([1])
                pair, layer_id = self.select_layer(layer_rand, cumulative_probabilities)
                t_layer_name = pair['target_layer']
                if current_params_list[t_layer_name]['kernel_amount'] >= 4: # (do while) it makes shure that it is at least 10 kernels in choosen layer
                    break

            # choosing the kernel
            current_kernels_amount = current_params_list[t_layer_name]['kernel_amount']
            kernel_id = np.random.randint(0, current_kernels_amount)

            # modifing params list
            current_params_list[t_layer_name]['kernel_amount'] -= 1
            current_params_list[t_layer_name]['params_amount'] -= current_params_list[t_layer_name]['params_per_kernel']
            all_params -= current_params_list[t_layer_name]['params_per_kernel']

            # removing kernel from model
            thh.remove_kernel(self.model, pair, kernel_id)
            ls.update_layer_pairs_after_removal(self.layers_pairs, pair_pruned=pair)
            kernels_counter += 1

            # reevaluating current flops ratio
            if current_flops_ratio - 0.05 > self.goal_flops_ratio: # if there is still lot to prune
                flops_update_timer -= 1
                if flops_update_timer == 0:
                    flops_update_timer = 20
                    flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
                    current_flops_ratio = flops_now / self.flops_origin
                    print("flops_ratio: ", current_flops_ratio)
                    # print(f"kernels pruned: {kernels_counter}")
            else: # if pruning aproaches the goal ratio updates are conducted every time
                    flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
                    current_flops_ratio = flops_now / self.flops_origin
                    print("flops_ratio: ", current_flops_ratio)
                    # print(f"kernels pruned: {kernels_counter}")
            
        # printing achieved flops reduction
        print( f"flops: {current_flops_ratio * 100}% | kernels removed: {kernels_counter}" )


class RandomRat:
    """!!!Need to be changed!!! This class implements random pruning algorithm. It is assures that there is at least 10 kernels in left.
    Random selection of kernels is weighted by the amount of parameters in layer to total sum of them in model.

    Attributes
    ----------
    model: torch.nn.Module
        model given to the class to be pruned
    goal_flops_ratio: float
        ratio of flops from original model to pruned model which algorithm i aimming to achieve
    layers_pairs: list[tuple[str]]
        pairs of layers as a list. It is used by algorithm no navigate through model
    input_shape: tuple[int]
        shape of images that are being used as input to the model
    """

    def __init__(self, goal_flops_ratio: float, model: th.nn.Module, layers_pairs: ls.LayerPairs):
        self.model = model
        self.goal_flops_ratio = goal_flops_ratio
        self.layers_pairs = copy.deepcopy(layers_pairs)
        self.input_shape = (1, 3, 224, 224)
        self.flops_origin = calflops.calculate_flops(self.model, self.input_shape, print_results=False, output_as_string=False)[0]


    def select_layer(self, x: float, cumulative_probabilities: th.Tensor) -> tuple[tuple[str], int]:
        """ This function returns weighted randomly selected layer.

        Arguments
        ----------
        x: float
            random sample from randge[0, 1] which will be projected on layer
        cumulative_probabilities: th.Tensor
            as probability here is number of paramters in layer by all paramters in all layers.
            This argument is cumulation of this probability distribution.
        """
        for i, prob in enumerate(cumulative_probabilities):
            if x <= prob:
                return (self.layers_pairs.get_by_id(i), i)


    def prune_model(self) -> None:
        """
        This functions conducts full pruning based on parameters given in constructor.
        It results with modification of given model.
        """
        current_params_list = dict()
        # all_params = 0
        all_kernels = 0

        for pair in self.layers_pairs:
            t_layer_name = pair['target_layer']

            kernel_amount = self.model.get_submodule(t_layer_name).weight.shape[0]
            all_kernels += kernel_amount

            current_params_list[t_layer_name] = {'kernel_amount' : int(kernel_amount)}

        flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
        current_flops_ratio = flops_now / self.flops_origin

        # pruning loop
        kernels_counter = 0
        flops_update_timer = 20
        while current_flops_ratio > self.goal_flops_ratio:

            # creating probabilities for weighting the random selection of layer
            probabilities = th.tensor([ current_params_list[pair['target_layer']]['kernel_amount'] / all_kernels for pair in self.layers_pairs ])
            cumulative_probabilities = th.cumsum(probabilities, dim=0)

            # choosing the layer
            while True:
                layer_rand = np.random.uniform(0, 1)
                pair, layer_id = self.select_layer(layer_rand, cumulative_probabilities)
                t_layer_name = pair['target_layer']
                if self.model.get_submodule(t_layer_name).weight.shape[0] >= 4: # (do while) it makes shure that it is at least one kernel in choosen layer
                    break
                
            # choosing the kernel
            current_kernels_amount = self.model.get_submodule(t_layer_name).weight.shape[0]
            kernel_id = np.random.randint(0, current_kernels_amount)
            current_params_list[t_layer_name]['kernel_amount'] -= 1
            all_kernels -= 1


            # removing kernel from model
            thh.remove_kernel(self.model, pair, kernel_id)
            ls.update_layer_pairs_after_removal(self.layers_pairs, pair_pruned=pair)
            kernels_counter += 1

            try:
                # reevaluating current flops ratio
                if current_flops_ratio - 0.05 > self.goal_flops_ratio: # if there is still lot to prune
                    flops_update_timer -= 1
                    if flops_update_timer == 0:
                        flops_update_timer = 20
                        flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
                        current_flops_ratio = flops_now / self.flops_origin
                        print("flops_ratio: ", current_flops_ratio)
                        # print(f"kernels pruned: {kernels_counter}")
                else: # if pruning aproaches the goal ratio updates are conducted every time
                        flops_now = calflops.calculate_flops(self.model, self.input_shape, output_as_string=False, print_results=False)[0]
                        current_flops_ratio = flops_now / self.flops_origin
                        print("flops_ratio: ", current_flops_ratio)
                        # print(f"kernels pruned: {kernels_counter}")
            except:
                print("pair: ", pair)
                print("kernel_id", kernel_id)
                print(self.model)
                raise Exception("?")
            

        # printing achieved flops reduction
        print( f"flops: {current_flops_ratio * 100}% | kernels removed: {kernels_counter}" )