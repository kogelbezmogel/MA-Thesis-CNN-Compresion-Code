import torch as th
import random

import sys
import copy
import calflops
import time

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
import torchhelper as thh
import LayerSchemes as ls


class Contribution():
    def __init__(
                    self,
                    goal_flops_ratio: float,
                    model:th.nn.Module,
                    layer_pairs: ls.LayerPairs,
                    train_dataloader: th.utils.data.DataLoader,
                    test_dataloader: th.utils.data.DataLoader,
                    hierarchical_groups: ls.HierarchicalGroups,
                    static_group: list,
                    retrain_epochs: int=2,
                    pruning_sample_size: int=32,
                    step_reduction_ratio: float=0.1,
                    flops_constraint: bool=True,
                    min_kernels_in_layer: int=8
                ):
        self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.hierarchical_groups = copy.deepcopy(hierarchical_groups)
        self.static_group = copy.deepcopy(static_group)
        self.static_ratio = 1/64
        self.step_reduction_ratio = step_reduction_ratio
        self.goal_flops_ratio = goal_flops_ratio
        self.layer_pairs = copy.deepcopy(layer_pairs)
        self.layer_pairs_copy = copy.deepcopy(layer_pairs)
        self.retrain_epochs = retrain_epochs
        self.pruning_sample_size = pruning_sample_size
        self.flops_constraint = flops_constraint
        self.min_kernels_in_layer = min_kernels_in_layer

        self.kernel_scores = dict()
        self.current_flops_ratio = 1.0
        self.origin_flops = calflops.calculate_flops(self.model, (1, 3, 224, 224), print_results=False, output_as_string=False)[0]

        # for hierarchical
        self.input_shapes = dict()
        self.output_shapes = dict()
        self.model_input_shape = (1, 3, 224, 224)

        self.all_layers = set()
        for pair in self.layer_pairs:
            self.all_layers.add( pair['target_layer'] )
            for follow in pair['follow_layers']:
                self.all_layers.add(follow)


    def reset_temp_attributes(self) -> None:
        """
        Add doc
        """
        self.kernel_scores = dict()
        # adding place for scores for each target_layer
        for pair in self.layer_pairs:
            target_layer_name = pair['target_layer']
            kernels_amount = self.model.get_submodule(target_layer_name).weight.shape[0]
            self.kernel_scores[target_layer_name] = th.zeros([kernels_amount], dtype=th.float64)

        # hooks for collecting shapes definition
        def get_input_output_shape(name):
            def hook(model, input, output):
                self.input_shapes[name] = tuple(input[0].shape)
                self.output_shapes[name] = tuple(output.shape)
            return hook

        # setting hooks
        hooks_handlers = []
        for layer_name in self.all_layers:
            handler = self.model.get_submodule(layer_name).register_forward_hook( get_input_output_shape(layer_name) )
            hooks_handlers.append( handler )
        random_input = th.empty(self.model_input_shape, dtype=th.float32).to(self.device)

        # updating input and output shapes for layers
        with th.no_grad():
            self.model.train()
            self.model.apply( thh.set_bn_eval )
            self.model(random_input)

        # removing hooks
        for handler in hooks_handlers:
            handler.remove()


    def create_hierarchical_pruning_schedule(self) -> None:
        """
        Add doc
        """
        # defining amount of kernels to prune in this iteration
        all_kernels = 0
        for group in self.hierarchical_groups.keys():
            for target_layer in self.hierarchical_groups[group]:
                all_kernels += self.model.get_submodule(target_layer).weight.shape[0]
                # print( f"{target_layer} : {self.model.get_submodule(target_layer).weight.shape[0]} | ", end='')
        pruning_amount = int(round(all_kernels*self.step_reduction_ratio, 0))

        # startgin with empty schedule
        self.hierarchical_pruning_schedule = dict()
        if self.flops_constraint:
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
                self.hierarchical_pruning_schedule[group] = int(round( flops_group[group]/flops_all*pruning_amount, 0 ))
        else:
            output_channels_all = 0
            group_channels = dict()
            for group in self.hierarchical_groups.keys():
                group_channels[group] = 0
                for t_layer_name in self.hierarchical_groups[group]:
                    layer = self.model.get_submodule(t_layer_name)
                    group_channels[group] += layer.out_channels
                    output_channels_all += layer.out_channels
            for group in self.hierarchical_groups.keys():
                self.hierarchical_pruning_schedule[group] = int(round( group_channels[group]/output_channels_all*pruning_amount, 0 ))

        # print(f"created hierarchy: {self.hierarchical_pruning_schedule}")
        # for group in self.hierarchical_pruning_schedule.keys():
        #     print( f"group {group} kernels: {self.hierarchical_pruning_schedule[group]}" )


    def get_geathering_scores_hook(self, target_layer_name: str, channels_slice_start: int, channels_slice_end: int):
        def hook(model, input):
            # calculating svd decomposition
            input_slice = input[0][:, channels_slice_start : channels_slice_end, :, :].detach().clone()
            try:
                S = th.linalg.svdvals(input_slice, driver='gesvdj')
            except th._C._LinAlgError:
                print("with noise")
                S = th.linalg.svdvals(input_slice + th.randn_like(input_slice) * 1e-8, driver='gesvdj')

            # summing along batch and diagonal elements for single matrix
            S = th.sum(S, dim=[0, 2])
            self.kernel_scores[target_layer_name] += S.detach().cpu().clone()
        return hook


    def add_hooks(self) -> list[th.utils.hooks.RemovableHandle]:
        """
        Add doc
        """
        # adding hooks to following_layer in layers_pairs
        hooks = []
        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            follow_layers_names = pair['follow_layers']
            channels_slice = [pair['coresponding_channels_slice']['start'], pair['coresponding_channels_slice']['end']]

            if 'substitution_follow_layer' in pair.keys():
                f_layer_name = pair['substitution_follow_layer']
                hook_handler = self.model.get_submodule(f_layer_name).register_forward_pre_hook( self.get_geathering_scores_hook(t_layer_name, channels_slice[0], channels_slice[1]) )
                hooks.append( hook_handler )
            else:
                for f_layer_name in follow_layers_names:
                    hook_handler = self.model.get_submodule(f_layer_name).register_forward_pre_hook( self.get_geathering_scores_hook(t_layer_name, channels_slice[0], channels_slice[1]) )
                    hooks.append( hook_handler )
        return hooks


    def remove_hooks(self, hooks: list[th.utils.hooks.RemovableHandle]) -> None:
        """
        Add doc
        """
        for handler in hooks:
            handler.remove()


    def prune_model(self) -> None:
        """
        Add doc
        """
        while self.current_flops_ratio > self.goal_flops_ratio:
            step_start = time.time()
            # restarting kernels scores and input output shapes
            self.reset_temp_attributes()
            self.model.apply( thh.set_bn_eval )

            # add hooks
            score_hooks = self.add_hooks()
            # passing 256 inputs through model to gather scores
            with th.no_grad():
                self.model = self.model.to(self.device)
                for step, (x, _) in enumerate(self.train_dataloader):
                    x = x.to(self.device)
                    self.model(x)
                    # stop iterating if sample demand is meet
                    if (step+1)*self.train_dataloader.batch_size >= self.pruning_sample_size:
                        break
            
            # hooks in this iteration won't be used anymore
            self.remove_hooks(score_hooks)

            # in paper results are not normalized in any way but here for the expanded version
            # when target layer has more than 1 follow layer this will cause higher scores for it
            # therefor to prevent that here scores are divided by num of follow layers
            for pair in self.layer_pairs:
                layer_name = pair['target_layer']
                norm_coef = len(pair['follow_layers'])
                self.kernel_scores[layer_name] /= norm_coef

            # creating hierarchy to define amounts of kernels to prune in each
            self.create_hierarchical_pruning_schedule()
 
            # removing
            self.remove_kernels_one_by_one()

            # testing accuracy
            print(f"model accuracy after removal: {thh.evaluate_model(self.model, self.test_dataloader):5.2f}")

            # printing removal
            for key in self.kernel_removal.keys():
                if self.kernel_removal[key] > 0:
                    print(f"{key:40s} -> {self.kernel_removal[key]:3d} kernels removed")
            self.update_current_flops_ratio()

            #finetuneing the model
            self.model.apply( thh.set_bn_train )
            self.model.requires_grad_(True)
            thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs)   
            self.model.apply( thh.set_bn_eval )         
            print( f"current flops_ratio after pruning: {round(self.current_flops_ratio, 4)}" )
            step_end = time.time()
            print( f"step time: {step_end-step_start:10.0f}" )
        print( f"model finished with flops_ratio: {round(self.current_flops_ratio, 4)}" )


    def update_current_flops_ratio(self):
        """
        Add doc
        """
        flops = calflops.calculate_flops(self.model, (1, 3, 224, 224), print_results=False, output_as_string=False)[0]
        self.current_flops_ratio = flops / self.origin_flops


    def remove_kernels_one_by_one(self):
        """
        Add doc
        """
        # this is the amount of kernels that needs to be left in single layer
        self.kernel_removal = dict()
        
        # removing from static list
        for t_layer_name in self.static_group:
            pair = self.layer_pairs[t_layer_name]
            # defining number of kernels to prune
            kernels_left = self.model.get_submodule(t_layer_name).weight.shape[0]
            to_prune = int(round( kernels_left * self.static_ratio ))
            max_amount = min(to_prune, kernels_left - self.min_kernels_in_layer)
            # finding and sorting the list of kernels to prune
            topk_res = th.topk(self.kernel_scores[t_layer_name], max_amount, largest=False)
            topk_res = topk_res.indices.tolist()
            topk_res.sort(reverse=True)
            for kernel_id in topk_res:
                thh.remove_kernel(self.model, self.layer_pairs[t_layer_name], kernel_id)
                ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)
                
        # iterating over each hierarchy
        for group in self.hierarchical_groups.keys():
            pruning_layers_list = dict()

            # gathering sorted list of 'max_amount' kernels and theirs scores sorted by the score
            for t_layer_name in self.hierarchical_groups[group]:
                self.kernel_removal[t_layer_name] = 0
                max_amount = min(self.hierarchical_pruning_schedule[group], self.model.get_submodule(t_layer_name).weight.shape[0] - self.min_kernels_in_layer) # if kenrels <= 6 then topk will be empty
                topk_res = th.topk(self.kernel_scores[t_layer_name], max_amount, largest=False)

                # copying because they are read only
                pruning_layers_list[t_layer_name] = dict()
                pruning_layers_list[t_layer_name]['values'] = topk_res.values
                pruning_layers_list[t_layer_name]['indices'] = topk_res.indices

            kernels_to_prune = self.hierarchical_pruning_schedule[group]
            kernels_pruned = 0
            flops_update_timer = 50
            # here can be error when there is single hierarchy group in which all layers have <= 5 kernels left. Then loop doesn't end
            while (kernels_pruned < kernels_to_prune) and (self.current_flops_ratio > self.goal_flops_ratio):
                layer_min = ''
                value_min = float('inf')

                # finding global minimum over all layers in group
                for t_layer_name in self.hierarchical_groups[group]:
                    if pruning_layers_list[t_layer_name]['values'].numel() != 0 and pruning_layers_list[t_layer_name]['values'][0] < value_min : # check if not empty
                        value_min = pruning_layers_list[t_layer_name]['values'][0]
                        layer_min = t_layer_name

                # if layer min has bean founded  the list must be altereted. If not kernels from this hierarchy cannot be deleted
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
                    # print(f"removing :{t_layer_name} -> {pair['follow_layers']} | {self.model.get_submodule()}")
                    thh.remove_kernel(self.model, pair, kernel_id)
                    ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)
                    self.kernel_removal[t_layer_name] += 1

                    # reevaluating current flops ratio
                    if self.current_flops_ratio - 0.05 > self.goal_flops_ratio: # if there is still lot to prune
                        flops_update_timer -= 1
                        if flops_update_timer == 0:
                            flops_update_timer = 30
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


    def test_criterion_vs_random(self) -> None:
        """
        Add doc
        """
        test_num = 10
        self.min_kernels_in_layer = 0

        layer_pairs_copy = copy.deepcopy(self.layer_pairs)
        h_groups_copy = copy.deepcopy(self.hierarchical_groups)
        model_copy = copy.deepcopy(self.model)

        # step reduction ratio will provide stopping point
        ratios = [ i/10 for i in range(1, 10) ]
        self.goal_flops_ratio = 0.0


        test_layers = []
        test_results = dict()
        for pair in self.layer_pairs:
            test_layers.append( pair['target_layer'] )
            test_results[ pair['target_layer'] ] = {'contr': [], 'rand': [] }

        for test_layer in test_layers:
            print(f"test_layer: {test_layer}")
            for ratio in ratios:
                self.step_reduction_ratio = ratio

                sum_of_results_rand = 0
                sum_of_results_contr = 0
                for test_id in range(test_num):

                    self.model = copy.deepcopy(model_copy)
                    self.update_current_flops_ratio()
                    self.layer_pairs = copy.deepcopy(layer_pairs_copy)
                    self.hierarchical_groups = copy.deepcopy(h_groups_copy)
                    # removing all diffrent layers than tested one
                    for layer in test_layers:
                        if layer != test_layer:
                            self.layer_pairs.remove(layer)
                            self.hierarchical_groups['group1'].remove(layer)
                    
                    # restarting kernels scores and input output shapes
                    self.reset_temp_attributes()
                    self.model.apply( thh.set_bn_eval )
                    # add hooks
                    score_hooks = self.add_hooks()
                    # passing 256 inputs through model to gather scores
                    with th.no_grad():
                        th.manual_seed( thh.get_random_seed() )
                        self.model = self.model.to(self.device)
                        for step, (x, _) in enumerate(self.train_dataloader):
                            x = x.to(self.device)
                            self.model(x)
                            # stop iterating if sample demand is meet
                            if (step+1)*self.train_dataloader.batch_size >= self.pruning_sample_size:
                                break
                    # hooks in this iteration won't be used anymore
                    self.remove_hooks(score_hooks)

                    # creating hierarchy to define amounts of kernels to prune in each
                    self.create_hierarchical_pruning_schedule() # it is created with step_reduction_ratio
                    model_random = copy.deepcopy(self.model)
                    
                    # removing till ratio limit is meet
                    self.remove_kernels_one_by_one()
                    acc_after_prune = thh.evaluate_model(self.model, self.test_dataloader)

                    # testing accuracy
                    pruned_report = dict()
                    for layer_name in self.kernel_removal.keys():
                        to_prune = self.kernel_removal[layer_name]
                        for i in range(to_prune):
                            if not layer_name in pruned_report.keys():
                                pruned_report[layer_name] = 1
                            else:
                                pruned_report[layer_name] += 1
                            kernel_rand = random.randint(0, self.model.get_submodule(layer_name).weight.shape[0])
                            pair = self.layer_pairs[layer_name]
                            thh.remove_kernel(model_random, pair, kernel_rand)

                    acc_after_random = thh.evaluate_model(model_random, self.test_dataloader)
                    sum_of_results_contr += acc_after_prune
                    sum_of_results_rand += acc_after_random

                print(f"ratio: {ratio} | cont={sum_of_results_contr / test_num:5.2f} rand={sum_of_results_rand / test_num:5.2f}")    
                test_results[test_layer]['contr'].append((ratio, sum_of_results_contr / test_num))
                test_results[test_layer]['rand'].append((ratio, sum_of_results_rand / test_num))
            print()
        return test_results 


    def sensitivity_analysis(self, portion_size: float=0.1):
        """
        Add doc
        """
        # results will be collected for each layer as a list of succesive reductions by portion_size of kernels.
        self.sensitivity_results = dict()
        self.min_kernels_in_layer = 0
        model_starting_point = copy.deepcopy(self.model)

        start = int(portion_size * 100)
        end = 100
        portion = int(100*portion_size)
        # it determines ratio and how many kernels will be removed at each point
        reductions = [val/100 for val in range(start, end, portion)]
        
        # starting eveluation
        start_acc = thh.evaluate_model(self.model, self.test_dataloader)

        for pair in self.layer_pairs:
            t_layer_name = pair['target_layer']
            f_layers_names = pair['follow_layers']
            print(f"{t_layer_name} analysis -------------------------------")
            layer_start = time.time()

            self.sensitivity_results[t_layer_name] = [(0.0, start_acc)]
            # setting number of kernels to prune for this layer
            all_kernels_num = self.model.get_submodule(t_layer_name).weight.shape[0]
            kernels_reductions = [ int(round(red*all_kernels_num, 0)) for red in reductions ]
        
            for i, reduction in enumerate(kernels_reductions):
                self.model.apply( thh.set_bn_eval )

                # scoring kernels
                with th.no_grad():
                    hook_handlers_svd = []
                    channels_slice_start, channels_slice_end = pair['coresponding_channels_slice']['start'], pair['coresponding_channels_slice']['end']
                    for f_layer_name in f_layers_names:
                        self.reset_temp_attributes()
                        hook_handler = self.model.get_submodule(f_layer_name).register_forward_pre_hook(self.get_geathering_scores_hook(t_layer_name, channels_slice_start, channels_slice_end))
                        hook_handlers_svd.append(hook_handler)

                        for step, (x, _) in enumerate(self.train_dataloader):
                            x = x.to(self.device)
                            self.model(x)
                            if (step+1)*self.train_dataloader.batch_size >= self.pruning_sample_size:
                                break

                    for hook_handler in hook_handlers_svd:
                        hook_handler.remove()

                # pruning the kernels
                kernel_list = list(th.topk(self.kernel_scores[t_layer_name], reduction, largest=False).indices)
                kernel_list.sort(reverse=True)
                for kernel_id in kernel_list:
                    thh.remove_kernel(self.model, pair, kernel_id)
                    # ls.update_layer_pairs_after_removal(self.layer_pairs, pair_pruned=pair)

                self.model.apply( thh.set_bn_train )
                # retraining the model to adjust weights before next kernels scoring
                thh.train_model(self.model, self.train_dataloader, self.test_dataloader, epochs=self.retrain_epochs, return_best_test=False, return_best_train=False)
                self.model.apply(thh.set_bn_eval)
                # saving result of reduction
                test_acc = thh.evaluate_model(self.model, self.test_dataloader)
                self.sensitivity_results[t_layer_name].append( (round(reductions[i], 2), round(test_acc, 5)) )
                
                # layer pairs need to be restarted in case when concat group is not empty
                # self.layer_pairs = copy.deepcopy(self.layer_pairs_copy)
                self.model.apply( thh.set_bn_eval )
                print(f"layer: {t_layer_name} | red: {reduction:4d} | {100*reductions[i]:5.0f}% | t_acc: {test_acc:5.2f} | t_layer: {self.model.get_submodule(t_layer_name).weight.shape}")
                # restarting the model
                self.model = copy.deepcopy(model_starting_point)

            layer_end = time.time()
            print(f"{t_layer_name} analysis: {round(layer_end-layer_start, 1)}s ---------------------------done\n")
        return self.sensitivity_results