import torch as th
import sys
import copy
import calflops
import time
import random

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls

class MeanGradient():
    def __init__(
                    self,
                    goal_flops_ratio: float,
                    model: th.nn.Module,
                    layer_pairs: ls.LayerPairs,
                    train_dataloader: th.utils.data.DataLoader,
                    test_dataloader: th.utils.data.DataLoader,
                    hierarchical_groups: ls.HierarchicalGroups,
                    static_group: ls.LayerPairs = list(),
                    retrain_epochs: int = 4,
                    n: int=100,
                    flops_constraint: bool=True,
                    min_kernels_in_layer: int=8
                ):
        """
        Add doc
        """
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.layer_pairs = copy.deepcopy(layer_pairs)
        self.layer_pairs_copy = copy.deepcopy(layer_pairs)
        self.hierarchical_groups = copy.deepcopy(hierarchical_groups)
        self.static_group = copy.deepcopy(static_group)
        self.static_ratio = 1/64 # from the papaer
        self.goal_flops_ratio = goal_flops_ratio
        self.retrain_epochs = retrain_epochs
        self.n = n
        self.flops_constrained=flops_constraint
        self.model_input_shape = (1, 3, 224, 224)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.min_kernels_in_layer = min_kernels_in_layer

        # this is initialized here as it reqiures access through whole alogrithm
        if type(model) == thh.GoogLeNetTAT:
            self.loss_fun = thh.googlenet_loss_train
        else:
            self.loss_fun = th.nn.CrossEntropyLoss()
        self.grad_hook_handlers = dict()
        self.pruning_hierarchy = None
        self.input_shapes = dict()
        self.output_shapes = dict()
        self.input_gradients_temp = dict()
        self.kernels_scores = dict()
        self.model = model.to(self.device)

        # atributes for saving and loading
        self.origin_flops = calflops.calculate_flops(self.model, self.model_input_shape, print_results=False, output_as_string=False)[0]
        self.current_flops_ratio = 1.0


    def __set_atributes_for_next_iteration(self) -> None:
        """
        This function prepares few atributes for iteration.
        Those are:
            inputs_shapes - because after pruning they don't match current model
            outputs_shapes - because after pruning they don't match current model
            gradient_sum - number of channels changes after pruning
            gradient_temp - after iteration those are redundant and it is big in memory
        """
        # hooks for collecting shapes definition
        def get_input_output_shape(name):
            def hook(model, input, output):
                self.input_shapes[name] = tuple(input[0].shape)
                self.output_shapes[name] = tuple(output.shape)
            return hook

        # setting hooks
        hooks_handlers = []
        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            handler = self.model.get_submodule(t_layer_name).register_forward_hook( get_input_output_shape(t_layer_name) )
            hooks_handlers.append( handler )
        random_input = th.empty(self.model_input_shape, dtype=th.float32).to(self.device)

        # updating input and output shapes for layers
        with th.no_grad():
            self.model(random_input)

        # removing hooks
        for handler in hooks_handlers:
            handler.remove()

        # updating gradient_sum by zeoring it out and changing lentgh according to channel_amount change
        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            channels_amount = self.output_shapes[t_layer_name][1]
            self.kernels_scores[t_layer_name] = th.zeros([channels_amount], dtype=th.float64)

        self.input_gradients_temp = dict()


    def __create_hierarchical_pruning_schedule(self) -> None:
        """ Pruning schedule is the essential varaible for pruning. It defines which layers will be pruned and by what number
        of kernels. 
        """
        # startgin with empty schedule
        self.hierarchical_pruning_schedule = dict()

        if self.flops_constrained:
            flops_all = 0
            flops_group = dict()
            for group in self.hierarchical_groups.keys():
                flops_group[group] = 0
                for t_layer_name in self.hierarchical_groups[group]:
                    input_shape = self.input_shapes[t_layer_name]
                    layer = self.model.get_submodule(t_layer_name)
                    flops = calflops.calculate_flops( layer, input_shape, print_results=False, output_as_string=False )[0]
                    flops_group[group] += flops
                    flops_all += flops
            for group in self.hierarchical_groups.keys():
                self.hierarchical_pruning_schedule[group] = int(round( flops_group[group]/flops_all*self.n, 0 ))
        else:
            output_channels_all = 0
            group_channels = dict()
            for group in self.hierarchical_groups.keys():
                group_channels[group] = 0
                for t_layer_name in self.hierarchical_groups[group]:
                    layer = self.model.get_submodule(t_layer_name)
                    group_channels[group] += layer.out_channels
                    output_channels_all += layer.out_channels
            if output_channels_all == 0:
                output_channels_all = 1e4
            for group in self.hierarchical_groups.keys():
                self.hierarchical_pruning_schedule[group] = int(round( group_channels[group]/output_channels_all*self.n, 0 ))


    def __add_grad_hooks(self, layer_pairs: list[tuple[str]]) -> None:
        """
        Add doc
        """
        # defining hook for collecting gradient in respect to layer input
        def get_grad_in_respect_to_input(name):
            def hook(model, input, output):
                self.input_gradients_temp[name] = output
                self.input_gradients_temp[name].retain_grad()
            return hook

        # adding hooks to layers
        for pair in layer_pairs:
            t_layer_name = pair['target_layer']
            self.grad_hook_handlers[t_layer_name] = self.model.get_submodule(t_layer_name).register_forward_hook( get_grad_in_respect_to_input(t_layer_name) )


    def __remove_grad_hooks(self) -> None:
        """
        Add doc
        """
        for key in self.grad_hook_handlers.keys():
            self.grad_hook_handlers[key].remove()
        self.grad_hook_handlers = dict()


    def __remove_kernels_one_by_one(self) -> None:
        # this is the amount of kernels that needs to be left in single layer

        # removing from static list
        for t_layer_name in self.static_group:
            pair = self.layer_pairs[t_layer_name]
            # defining number of kernels to prune
            kernels_left = self.model.get_submodule(t_layer_name).weight.shape[0]
            to_prune = int(round( kernels_left * self.static_ratio ))
            max_amount = min(to_prune, kernels_left - self.min_kernels_in_layer)
            # finding and sorting the list of kernels to prune
            topk_res = th.topk(self.kernels_scores[t_layer_name], max_amount, largest=False)
            topk_res = topk_res.indices.tolist()
            topk_res.sort(reverse=True)
            for kernel_id in topk_res:
                thh.remove_kernel(self.model, self.layer_pairs[t_layer_name], kernel_id)
                ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)

        self.update_current_flops_ratio()

        self.kernel_removal = dict()
        # removing = []
        # iterating over each hierarchy
        flops_update_timer = 20
        for group in self.hierarchical_groups.keys():
            pruning_layers_list = dict()

            # gathering sorted list of 'max_amount' kernels and theirs scores sorted by the score
            for t_layer_name in self.hierarchical_groups[group]:
                self.kernel_removal[t_layer_name] = 0
                max_amount = min(self.hierarchical_pruning_schedule[group], self.model.get_submodule(t_layer_name).weight.shape[0] - self.min_kernels_in_layer) # if kenrels <= 5 then topk will be empty
                topk_res = th.topk(self.kernels_scores[t_layer_name], max_amount, largest=False)

                # copying because they are read only
                pruning_layers_list[t_layer_name] = dict()
                pruning_layers_list[t_layer_name]['values'] = topk_res.values
                pruning_layers_list[t_layer_name]['indices'] = topk_res.indices

            kernels_to_prune = self.hierarchical_pruning_schedule[group]
            kernels_pruned = 0
            # here can be error when there is single hierarchy group in which all layers have <= 5 kernels left. Then loop doesn't end
            while kernels_pruned < kernels_to_prune and self.current_flops_ratio > self.goal_flops_ratio:
                layer_min = ''
                value_min = float('inf')

                # finding global minimum over all layers in group
                for t_layer_name in self.hierarchical_groups[group]:
                    if pruning_layers_list[t_layer_name]['values'].numel() != 0 and pruning_layers_list[t_layer_name]['values'][0] < value_min : # check if not empty
                        value_min = pruning_layers_list[t_layer_name]['values'][0]
                        layer_min = t_layer_name

                # if layer min has bean founded the list must be altereted. If not kernels from this hierarchy cannot be deleted
                if layer_min != '': 
                    # reindexing kernels with higher id than pruned kernel
                    kernel_id = pruning_layers_list[layer_min]['indices'][0]
                    kernels_to_reindex = pruning_layers_list[layer_min]['indices'] > kernel_id
                    pruning_layers_list[layer_min]['indices'][kernels_to_reindex] -= 1

                    # removing value and idice coresponding to founded min
                    pruning_layers_list[layer_min]['indices'] = pruning_layers_list[layer_min]['indices'][1:]
                    pruning_layers_list[layer_min]['values'] = pruning_layers_list[layer_min]['values'][1:]

                    # removing kernel from model
                    pair = list(filter(lambda pair: pair['target_layer']==layer_min, self.layer_pairs))[0]
                    t_layer_name = pair['target_layer']
                    # print("h: ", t_layer_name, pair['follow_layers'])
                    thh.remove_kernel(self.model, pair, kernel_id)
                    ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)
                    self.kernel_removal[t_layer_name] += 1
                    # removing.append(kernel_id.item())

                    # reevaluating current flops ratio
                    if self.current_flops_ratio - 0.05 > self.goal_flops_ratio: # if there is still lot to prune
                        flops_update_timer -= 1
                        if flops_update_timer == 0:
                            flops_update_timer = 20
                            self.update_current_flops_ratio()
                    else: # if pruning aproaches the goal ratio updates are conducted every time
                        self.update_current_flops_ratio()
                kernels_pruned += 1 # to end loop even if no kernels are pruned due to min_kernel_in_layer constraint
        
        # checking all layer in layer_pairs. The ones that cannot be further pruned because of 
        # min_kernels_in_layer constraint will be removed from layer_pairs and hierarchical_groups to avoid adding hooks to them.
        for group in self.hierarchical_groups.keys():
            for t_layer_name in self.hierarchical_groups[group]:
                current_kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]
                if current_kernels_num <= self.min_kernels_in_layer:
                    self.hierarchical_groups[group].remove(t_layer_name)
                    self.layer_pairs.remove(t_layer_name)
                    print(f"{t_layer_name} removed from list as min_kernels was reached")
        for t_layer_name in self.static_group:
            current_kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]
            if current_kernels_num <= self.min_kernels_in_layer:
                self.static_group.remove(t_layer_name)
                self.layer_pairs.remove(t_layer_name)
                print(f"{t_layer_name} removed from list as min_kernels was reached")


    def update_current_flops_ratio(self) -> None:
        current_flops = calflops.calculate_flops(self.model, (1, 3, 224, 224), print_results=False, output_as_string=False)[0]
        self.current_flops_ratio = current_flops / self.origin_flops


    def prune_model(self) -> None:
        """
        Add doc
        """
        print( f"\nstart | flops ratio: {self.current_flops_ratio*100: 3.1f}% -> {self.goal_flops_ratio*100: 3.1f}%" )

        # main pruning loop
        self.model.apply( thh.set_bn_eval )
        while self.current_flops_ratio > self.goal_flops_ratio:
            step_start = time.time()
            self.__set_atributes_for_next_iteration()
            self.__create_hierarchical_pruning_schedule() 
            self.__gather_gradient_scores()
            
            # removing self.n kernels or some part of self.n until goal_flops_ratio is reached
            self.__remove_kernels_one_by_one()

            # testing accuracy
            print(f"model accuracy after removal: {thh.evaluate_model(self.model, self.test_dataloader):5.2f}")

            # printing removal
            for key in self.kernel_removal.keys():
                if self.kernel_removal[key] > 0:
                    print(f"{key:40s} -> {self.kernel_removal[key]:3d} kernels removed")

            print( f"kernels removed. current flops: {self.current_flops_ratio * 100:5.2f}%")
                
            self.model.apply( thh.set_bn_train )
            thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs)
            self.model.apply( thh.set_bn_eval )
            step_end = time.time()
            print(f"step time: {step_end-step_start:10.0f}s")
        print( f"end | pruning and fine tuning done. current flops: {self.current_flops_ratio * 100:5.2f}%")


    def test_criterion_vs_random(self) -> None:
        """
        Add doc
        """
        test_num = 2
        model_restart = copy.deepcopy(self.model)
        layer_pairs_restart = copy.deepcopy(self.layer_pairs)
        h_groups_restart = copy.deepcopy(self.hierarchical_groups)

        self.goal_flops_ratio=0.0
        ratios = [i/10 for i in range(3, 7)]

        test_results = dict()
        test_layers = []
        for pair in self.layer_pairs:
            test_layers.append( pair['target_layer'] )
            test_results[ pair['target_layer'] ] = {"mean" : [], "rand" : []}

        for test_layer in test_layers:
            print(f"test_layer: {test_layer}")

            for ratio in ratios:             
                sum_acc_rand = 0
                sum_acc_mean = 0
            
                for test_id in range(test_num):
                    th.manual_seed( thh.get_random_seed() )
                    self.model = copy.deepcopy(model_restart)
                    self.layer_pairs = copy.deepcopy(layer_pairs_restart)
                    self.hierarchical_groups = copy.deepcopy(h_groups_restart)
                    for layer in test_layers:
                        if layer != test_layer:
                            self.layer_pairs.remove(layer)
                            self.hierarchical_groups['group1'].remove(layer)
                    to_prune = int(round(ratio * self.model.get_submodule(test_layer).weight.shape[0], 0))
                    self.n = to_prune   

                    self.model.apply( thh.set_bn_eval )
                    self.__set_atributes_for_next_iteration()
                    self.__create_hierarchical_pruning_schedule() 
                    self.__gather_gradient_scores()
                    model_rand = copy.deepcopy(self.model)    

                    # removing self.n kernels or some part of self.n until goal_flops_ratio is reached
                    self.__remove_kernels_one_by_one()
                    self.layer_pairs = copy.deepcopy(layer_pairs_restart)

                    for i in range(to_prune):
                        kernel_id = random.randint(0, model_rand.get_submodule(test_layer).weight.shape[0]-1)
                        thh.remove_kernel(model_rand, self.layer_pairs[test_layer], kernel_id)
                        ls.update_layer_pairs_after_removal(self.layer_pairs, pair)
                    acc_mean = thh.evaluate_model(self.model, self.test_dataloader)
                    acc_rand = thh.evaluate_model(model_rand, self.test_dataloader)
                    print(f"test{test_id}: mean={acc_mean:5.2f} rand={acc_rand:5.2f}")
                    sum_acc_mean += acc_mean
                    sum_acc_rand += acc_rand

                test_results[test_layer]['mean'].append((ratio, sum_acc_mean / test_num))
                test_results[test_layer]['rand'].append((ratio, sum_acc_rand / test_num))
                print(f"Summary: {ratio} | pruned: {to_prune} | mean={test_results[test_layer]['mean'][-1][1]:5.2f} rand={test_results[test_layer]['rand'][-1][1]:5.2f}\n\n")


    def __gather_gradient_scores(self, layer_pairs: ls.LayerPairs = None) -> None:
        """
        Arguments
        ---------
        layer_pairs: list[tuple[str]]
            if gathering gradients need to conducted on specific layers it is achivable by this argument.
        """
        if layer_pairs == None:
            layer_pairs = self.layer_pairs

        self.__add_grad_hooks(layer_pairs)
        self.model.requires_grad_(False)
        samples_counter = 0

        self.model.train()
        for step, (x, y_true) in enumerate(self.train_dataloader): # batch_size should be smaller maybe 64 instead of 256

            x, y_true = x.to(self.device), y_true.to(self.device)
            x.requires_grad = True

            y_pred = self.model(x) # this will trigger all hooks
            loss = self.loss_fun(y_pred, y_true)
            # gradients are being saved to input_gradients_temp with hooks
            loss.backward()

            for pair in layer_pairs:
                t_layer_name = pair['target_layer']
                
                feature_map_shape = th.tensor(self.input_gradients_temp[t_layer_name].shape[-2 :])
                # suming along feature_map dimensions
                temp_sum = th.sum(self.input_gradients_temp[t_layer_name].grad, dim=[2, 3])
                # normalizing the sum
                temp_sum = th.divide(temp_sum, th.prod(feature_map_shape))
                # getting absolute before summing along batch dimension
                temp_sum = th.abs(temp_sum)
                # summing along batch dimension
                temp_sum = th.sum(temp_sum, dim=[0])                
                self.kernels_scores[t_layer_name] += temp_sum.detach().cpu().clone()

            self.input_gradients_temp = dict()
            samples_counter += x.shape[0]

        # normalization to [0,1] gradients scores in each layer
        for pair in layer_pairs:
            t_layer_name = pair['target_layer']
            self.kernels_scores[t_layer_name] /= th.sqrt( th.pow(self.kernels_scores[t_layer_name], 2).sum() )

        self.model = self.model.cpu()
        self.__remove_grad_hooks()
        self.model.requires_grad_(True)


    def sensitivity_analysis(self, portion_size: float=0.1):
        # results will be collected for each layer as a list of succesive reductions by portion_size of kernels.
        self.sensitivity_results = dict()
        self.model.apply( thh.set_bn_eval )
        model_restart = copy.deepcopy(self.model)

        # starting eveluation
        start_acc = thh.evaluate_model(self.model, self.test_dataloader)

        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            f_layers_names = pair['follow_layers']
            optional_layers = pair['optional']
            print(f"{t_layer_name} analysis -------------------------------")
            
            layer_start = time.time()
            self.sensitivity_results[t_layer_name] = [(0.0, start_acc)]
            all_kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]

            start = int(portion_size * 100)
            end = 100
            portion = int(100*portion_size)
            # it determines ratio and how many kernels will be removed at each point
            reductions = [val/100 for val in range(start, end, portion)]
            kernels_reductions = [ int(round(red*all_kernels_num, 0)) for red in reductions ]
            # for each 10% of layer kernels layer will be addtionaly retrained by 1 epoch
            # sensivity analysis is not described fully in paper so this is assumption

            for i, red_num in enumerate(kernels_reductions):
                self.__set_atributes_for_next_iteration()
                
                artificial_layer_pairs = ls.LayerPairs()
                artificial_layer_pairs.append({
                    "target_layer" : t_layer_name,
                    "follow_layers" : f_layers_names,
                    "optional" : optional_layers
                })
                self.__gather_gradient_scores(artificial_layer_pairs)

                kernels_list = list(th.topk(self.kernels_scores[t_layer_name], red_num, largest=False).indices)
                kernels_list.sort(reverse=True)
                for kernel_id in kernels_list:
                    thh.remove_kernel(self.model, pair, kernel_id)

                # retrainig
                self.model.apply( thh.set_bn_train )
                thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs, return_best_test=False, return_best_train=False)
                self.model.apply( thh.set_bn_eval )

                test_acc = thh.evaluate_model(self.model, self.test_dataloader)
                print(f"done reduction: {reductions[i]:5.2f} | red_num: {red_num:3d} | test_acc: {test_acc:5.2f} | {self.model.get_submodule(t_layer_name).weight.shape}")
                self.sensitivity_results[t_layer_name].append( (round(reductions[i], 2), round(test_acc, 3)) )
                
                self.model = copy.deepcopy(model_restart)
            layer_end = time.time()
            print(f"{t_layer_name} analysis: {round(layer_end-layer_start, 1)}s ---------------------------done\n")