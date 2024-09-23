import torchhelper as thh
import torch as th
import copy


################################################################# INFO ####################
''' In this module functions return two types of output:
    
1)
    layer_pairs: list[dict[str]] - this is general way of listing all layers to prune the model. Data scheme can be seen below.
    layer_pairs = [
        {
            "target_layer" : "t_name_1",                                                                   - layer to prune
            "follow_layers" : [ "f_name_1", "f_name_2", ... ],                                             - all layers to which signal is transoported from target_layer
            "optional" : [ "batchnorm_or_parallelconv_name_1", "batchnorm_or_parallelconv_name_2", ...],   - all layers along the way of signal to follow layers
            "coresponding_channels_slice": {                                                               - it defines slice of input to follow layers which was produced
        },                                  "strat": int,                                                    only by target layer. It is important for googlenet and densenet as                                                        
                                            "end": int,                                                      they use concatenation of channels from multiple layers.
                                            "concatenation_group": []
                                           }
        {                                                                                         

            "target_layer" : "t_name_2",
            "follow_layers" : ...,
            ...
        },
        etc.
    ]

2) 
    hierarchical_groups: dict[list[dict[str]]]
    hierarchical_groups = {
        "group1" : { "pairs" : [
            {
                "target_layer" : "t_name_1",                                                                   - layer to prune
                "follow_layers" : [ "f_name_1", "f_name_2", ... ],                                             - all layers to which signal is transoported from target_layer
                "optional" : [ "batchnorm_or_parallelconv_name_1", "batchnorm_or_parallelconv_name_2", ...],   - all layers along the way of signal to follow layers
                "coresponding_channels_slice": [ strat: int, end: int ]                                        - it defines slice of input to follow layers
            },
            ... (another layer_pairs in this group) ...            
        ]},
        "group2" : { "pairs" : [
            ... (all layer_pairs in this group) ...
        ]},
        etc.
    }
'''


class HierarchicalGroups(dict):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __str__(self) -> str:
        representation = f"hierarchical groups {self.name}: \n"
        # for group in self.keys():
        #     pairs = self[group]['pairs']
        #     representation += f"{group}\n"
        #     for i, pair in enumerate(pairs):
        #         representation += f"    {i:2d} | {pair}\n"
        return representation


class LayerPairs():
    def __init__(self, tab: list= [], name: str="No name"):
        self.name = name
        self.tab = copy.deepcopy(tab)
        self.keys_removed = dict()
        self.hash_tab = self.create_hash_tab()
        
    def __iter__(self):
        return self.tab.__iter__()

    def __str__(self) -> str:
        representation = f"layer pairs set {self.name}: \n"
        for i, pair in enumerate(self):
            representation += f"{i:2d} | {pair['target_layer']}\n"
            representation += f"    follow| {pair['follow_layers']} \n"
            representation += f"    slice | {pair['coresponding_channels_slice']} \n"
        return representation
    
    def __getitem__(self, name: str):
        return self.tab[ self.hash_tab[name] ]

    def pop(self, i: int):
        self.keys_removed[ self.tab[i]['target_layer'] ] = True
        self.tab.pop(i)
        self.hash_tab = self.create_hash_tab()

    def remove(self, t_layer_name: str):
        self.keys_removed[t_layer_name] = True
        pair = self.tab[ self.hash_tab[t_layer_name] ]
        self.tab.remove(pair)
        self.hash_tab = self.create_hash_tab()
    
    def append(self, pair):
        self.tab.append(pair)
        self.hash_tab[ pair['target_layer'] ] = len(self.tab)-1

    def get_by_id(self, i: int):
        return self.tab[i]
    
    def was_removed(self, t_layer_name:str):
        if self.keys_removed[t_layer_name] == True:
            return True
        else:
            return False
    
    def create_hash_tab(self) -> dict:
        hash_tab = dict()
        for pair_id, pair in enumerate(self.tab):
            hash_tab[ pair['target_layer'] ] = pair_id
        return hash_tab


    
def update_layer_pairs_after_removal(layer_pairs: LayerPairs, pair_pruned: dict):
    concat_group = pair_pruned['coresponding_channels_slice']['concatenation_group']
    target_layer = pair_pruned['target_layer']
    # shifting end from target
    layer_pairs[target_layer]['coresponding_channels_slice']['end'] -= 1
    
    # shifting all slices from concat group
    for layer_name in concat_group:
        try:
            layer_pairs[layer_name]['coresponding_channels_slice']['start'] -= 1
            layer_pairs[layer_name]['coresponding_channels_slice']['end'] -= 1
        except Exception as e:
            if not layer_pairs.was_removed(layer_name):
                raise Exception( e.__str__() )
    


################################################################## ALEXNET ############################################
def get_channels_slice_alexnet(model: th.nn.Module, target_layer_name: str, follow_layers_names: list[str]) -> list[int, int]:
    channels_produced_by_target_layer_num = model.get_submodule(target_layer_name).weight.shape[0]
    return {
                "start": 0,
                "end": channels_produced_by_target_layer_num,
                "concatenation_group" : []
            }

def get_layer_pairs_alexnet() -> LayerPairs:
    model = thh.AlexNetGAP()
    layer_pairs = LayerPairs(name="LP | AlexNet")
    for t_id, f_id in [(0, 3), (3, 6), (6, 8), (8, 10)]:
        layer_pairs.append({
            "target_layer" : f"features.{t_id}",
            "follow_layers" : [f'features.{f_id}'],
            "optional" : [],
            "coresponding_channels_slice" : get_channels_slice_alexnet(model, f"features.{t_id}", [f'features.{f_id}'])
            })
    layer_pairs.append({"target_layer" : f"features.10",
                        "follow_layers" : [f'classifier.0'],
                        "optional" : [],
                        "coresponding_channels_slice" : get_channels_slice_alexnet(model, f'features.10', [f'classifier.0'])
                        })
    return layer_pairs


def get_hierarchical_groups_alexnet_1h() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 1 hierarchy | global pruning")
    hierarchical_groups['group1'] = []

    for t_id, f_id in [(0, 3), (3, 6), (6, 8), (8, 10)]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
            
    hierarchical_groups['group1'].append(f'features.10')
    return hierarchical_groups


def get_hierarchical_groups_alexnet_2h_v1() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 2 hierarchies | according to sensitivity analysis of Contribution")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []

    for t_id in [0, 3, 6, 8]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    hierarchical_groups['group2'].append(f'features.10')
    return hierarchical_groups


def get_hierarchical_groups_alexnet_3h_v1() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 3 hierarchies | according to sensitivity analysis MeanGradient")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []

    for t_id in [0]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [3, 6, 8]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    for t_id in [10]:
        hierarchical_groups['group3'].append(f'features.{t_id}')
    return hierarchical_groups


def get_hierarchical_groups_alexnet_2h_v2_contr() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 2 hierarchies | according to sensitivity analysis of Contribution")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []

    for t_id in [0, 3]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [6, 8, 10]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    return hierarchical_groups


def get_hierarchical_groups_alexnet_2h_v2_mean() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 2 hierarchies | according to sensitivity analysis of MeanGradient")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []

    for t_id in [0, 3, 6]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [8, 10]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    return hierarchical_groups

def get_hierarchical_groups_alexnet_2h_v3_mean() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 2 hierarchies | according to sensitivity analysis of MeanGradient")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []

    for t_id in [0]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [3, 6, 8, 10]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    return hierarchical_groups

def get_hierarchical_groups_alexnet_3h_v2_mean() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | AlexNet | 2 hierarchies | according to sensitivity analysis of MeanGradient")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []

    for t_id in [0, 3]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [6]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [8, 10]:
        hierarchical_groups['group3'].append(f'features.{t_id}')
    return hierarchical_groups


################################################################## VGG #################################################
def get_channels_slice_vgg(model: th.nn.Module, target_layer_name: str, follow_layers_names: list[str]) -> list[int, int]:
    channels_produced_by_target_layer_num = model.get_submodule(target_layer_name).weight.shape[0]
    return {
                "start": 0,
                "end": channels_produced_by_target_layer_num,
                "concatenation_group" : []
            }

vgg_conv_ids = [
                (0, 2),
                (2, 5),
                (5, 7),
                (7, 10),
                (10, 12),
                (12, 14),
                (14, 17),
                (17, 19),
                (19, 21),
                (21, 24),
                (24, 26),
                (26, 28)
               ]

def get_layer_pairs_vgg() -> LayerPairs:
    model = thh.Vgg16GAP()
    layer_pairs = LayerPairs(name="LP | VGG16")
    for t_id, f_id in vgg_conv_ids:
        layer_pairs.append({
            "target_layer" : f"features.{t_id}",
            "follow_layers" : [f'features.{f_id}'],
            "optional" : [],
            "coresponding_channels_slice" : get_channels_slice_alexnet(model, f"features.{t_id}", [f'features.{f_id}'])
            })
    layer_pairs.append({"target_layer" : f"features.28",
                        "follow_layers" : [f'classifier.0'],
                        "optional" : [],
                        "coresponding_channels_slice" : get_channels_slice_alexnet(model, f"features.28", [f'classifier.0'])
                        })
    return layer_pairs


def get_hierarchical_groups_vgg_1h() -> HierarchicalGroups:
    model = thh.Vgg16GAP()
    hierarchical_groups = HierarchicalGroups("HG | VGG16 | global pruning")
    hierarchical_groups['group1'] = []

    for t_id, f_id in vgg_conv_ids:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    hierarchical_groups['group1'].append(f'features.28')
    return hierarchical_groups


def get_hierarchical_groups_vgg_4h_contr() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | VGG16 | global pruning | according to Contribution sens")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []

    for t_id in [0]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [2, 5, 7, 10, 12, 14]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    for t_id in [17, 19, 21, 28]:
        hierarchical_groups['group3'].append(f'features.{t_id}')
    for t_id in [24, 26]:
        hierarchical_groups['group4'].append(f'features.{t_id}')
    return hierarchical_groups
    
    
def get_hierarchical_groups_vgg_4h_meang() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | VGG16 | global pruning | according to MeanGrad sens")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []

    for t_id in [0, 14]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [21, 17, 24]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    for t_id in [2, 10, 12]:
        hierarchical_groups['group3'].append(f'features.{t_id}')
    for t_id in [5, 7, 19, 26, 28]:
        hierarchical_groups['group4'].append(f'features.{t_id}')
    return hierarchical_groups


def get_hierarchical_groups_vgg_3h_meang_2at() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | VGG16 | 3h")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []

    for t_id in [0, 2, 5, 7]:
        hierarchical_groups['group1'].append(f'features.{t_id}')
    for t_id in [10, 12, 14]:
        hierarchical_groups['group2'].append(f'features.{t_id}')
    for t_id in [17, 19, 21, 24, 26, 28]:
        hierarchical_groups['group3'].append(f'features.{t_id}')
    return hierarchical_groups


################################################################## RESNET ##############################################
def get_channels_slice_resnet(model: th.nn.Module, target_layer_name: str, follow_layers_names: list[str]) -> list[int, int]:
    channels_produced_by_target_layer_num = model.get_submodule(target_layer_name).weight.shape[0]
    return {
                "start": 0,
                "end": channels_produced_by_target_layer_num,
                "concatenation_group" : []
            }



def get_static_group_resnet50() -> LayerPairs:
    model = thh.ResNet50T()
    static_group = []#LayerPairs(name="SG | resnet50 | frist convblock from each hierarchy and all last convlayers from convblock")
    for layer_id in range(1, 5):   
        # first two conv layers from first block
        block_id = 0

        for i in range(1, 3):
            static_group.append(f'layer{layer_id}.{block_id}.conv{i}')
        # last conv layer from first block (the one with down-sample)
        static_group.append(f'layer{layer_id}.{block_id}.conv{3}')

        # all blocks last conv layer
        for block_id in range(1, len(model.get_submodule(f'layer{layer_id}'))-1 ):
            conv_id = 3
            static_group.append(f'layer{layer_id}.{block_id}.conv{3}')
            
        # last block from layer_id
        block_id = len(model.get_submodule(f'layer{layer_id}'))-1
        conv_id = 3
        if layer_id != 4:
            static_group.append(f'layer{layer_id}.{block_id}.conv{3}')
        else:
            static_group.append(f'layer{layer_id}.{block_id}.conv{conv_id}')
    del model
    return static_group


def get_hierarchical_groups_4h_resnet50_global_no_static() -> HierarchicalGroups:
    model = thh.ResNet50T()
    hierarchical_groups = HierarchicalGroups("hierarchical groups | resnet50 | 4 hierarchies")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []

    for group in hierarchical_groups.keys():
        layer_id = int(group[-1])

        for block_id in range( 0, len(model.get_submodule(f'layer{layer_id}')) ): # from every block
            for conv_id in range(1, 3): # first two layers from each block 
                hierarchical_groups[group].append(f'layer{layer_id}.{block_id}.conv{conv_id}')
    del model
    return hierarchical_groups


def get_hierarchical_groups_4h_resnet50() -> HierarchicalGroups:
    model = thh.ResNet50T()
    hierarchical_groups = HierarchicalGroups("hierarchical groups | resnet50 | 4 hierarchies")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []

    for group in hierarchical_groups.keys():
        layer_id = int(group[-1])

        for block_id in range( 1, len(model.get_submodule(f'layer{layer_id}')) ): # from every block except first one
            for conv_id in range(1, 3): # first two layers from each block 
                hierarchical_groups[group].append(f'layer{layer_id}.{block_id}.conv{conv_id}')
    del model
    return hierarchical_groups


def get_hierarchical_groups_1h_resnet50() -> HierarchicalGroups:
    model = thh.ResNet50T()
    hierarchical_groups = HierarchicalGroups("hierarchical groups | resnet50 | 1 hierarchy | (global pruning)")
    hierarchical_groups['group1'] = []
    # starting_block = 0 if include_first_block else 1

    for group in hierarchical_groups.keys():
        for layer_id in range(5):
            for block_id in range( 1, len(model.get_submodule(f'layer{layer_id}')) ): # from every block
                for conv_id in range(1, 3): # first two layers from each block 
                    hierarchical_groups[group].append(f'layer{layer_id}.{block_id}.conv{conv_id}')
    del model
    return hierarchical_groups


def get_layer_pairs_resnet50_experimental() -> LayerPairs:
    model = thh.ResNet50T()
    layer_pairs = LayerPairs(name="all layer pairs | resnet50 | hierarchies + static_group")

    for layer_id in range(1, 5):   
        # first two conv layers from first block
        block_id = 0
        for conv_id in range(1, 3):
            layer_pairs.append({
                'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',       
                'follow_layers' : [f'layer{layer_id}.{block_id}.conv{conv_id+1}'],     
                'optional' : [  f'layer{layer_id}.{block_id}.bn{conv_id}' ],
                "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id}.{block_id}.conv{conv_id+1}'])
            })
        # last conv layer from first block (the one with down-sample)
        # if layer_id != 4:
        parallel = [f'layer{layer_id}.{block_id}.downsample.0']
        # else:
            # parallel = []     
        layer_pairs.append({
            'target_layer' : f'layer{layer_id}.{block_id}.conv{3}',
            'follow_layers' : [f'layer{layer_id}.{block_id+1}.conv{1}'],
            'omitted_parallel_layers' : parallel, # here for resnet projection kernels
            'optional' : [  f'layer{layer_id}.{block_id}.bn{3}', f'layer{layer_id}.{block_id}.downsample.1' ],
            "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{3}', [f'layer{layer_id}.{block_id+1}.conv{1}'])
        })

        # all middle blocks conv layers
        for block_id in range(1, len(model.get_submodule(f'layer{layer_id}'))-1 ):
            for conv_id in range(1, 3): # frist two layers
                layer_pairs.append({
                    'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',       
                    'follow_layers' : [f'layer{layer_id}.{block_id}.conv{conv_id+1}'],     
                    'optional' : [f'layer{layer_id}.{block_id}.bn{conv_id}'],
                    "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id}.{block_id}.conv{conv_id+1}'])
                })
            conv_id = 3 # last residual layer
            layer_pairs.append({
                    'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',
                    'follow_layers' : [f'layer{layer_id}.{block_id+1}.conv{1}'],
                    'optional' : [ f'layer{layer_id}.{block_id}.bn{conv_id}' ],
                    "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id}.{block_id+1}.conv{0}'])
                    })
            
        # last block from layer_id
        block_id = len(model.get_submodule(f'layer{layer_id}'))-1
        for conv_id in range(1, 3): # first two layers
            layer_pairs.append({
                'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',       
                'follow_layers' : [f'layer{layer_id}.{block_id}.conv{conv_id+1}'],     
                'optional' : [f'layer{layer_id}.{block_id}.bn{conv_id}'],
                "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id}.{block_id}.conv{conv_id+1}'])
            })
        conv_id = 3
        if layer_id != 4:
            layer_pairs.append({
                    'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',
                    'follow_layers' : [f'layer{layer_id+1}.{0}.conv{1}'],
                    'omitted_follow_layers' : [ f'layer{layer_id+1}.{0}.downsample.0' ], # here for resnet projection channels
                    'optional' : [f'layer{layer_id}.{block_id}.bn{conv_id}'],
                    "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id+1}.{0}.conv{0}'])
                    })
        else:
            layer_pairs.append({
                    'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',
                    'follow_layers' : [f'fc'],
                    'substitution_follow_layer' :  'avgpool', # here for contribution need to be substitute follow layer for criterion
                    'optional' : [f'layer{layer_id}.{block_id}.bn{conv_id}'],
                    "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'fc'])
                    })
    del model
    return layer_pairs


def get_layer_pairs_resnet50_classic() -> LayerPairs:
    model = thh.ResNet50T()
    layer_pairs = LayerPairs(name="LP | resnet50 | only first two convlayers from each resblock | (classic)")

    for layer_id in range(1, 5):   
        for block_id in range( 0, len(model.get_submodule(f'layer{layer_id}')) ): # from every block except first one
            for conv_id in range(1, 3): # first two layers from each block 
                layer_pairs.append({
                    'target_layer' : f'layer{layer_id}.{block_id}.conv{conv_id}',
                    'follow_layers' : [f'layer{layer_id}.{block_id}.conv{conv_id+1}'],
                    'optional' : [f'layer{layer_id}.{block_id}.bn{conv_id}'],
                    "coresponding_channels_slice" : get_channels_slice_resnet(model, f'layer{layer_id}.{block_id}.conv{conv_id}', [f'layer{layer_id}.{block_id}.conv{conv_id+1}'])
                })
    del model
    return layer_pairs


################################################################## DENSENET ##############################################
def get_channels_slice_densenet(model: th.nn.Module, target_layer_name: str, follow_layers_names: list[str]) -> list[int, int]:
    smallest_following_layer_name = follow_layers_names[0] # finding smallest layer in case they are not in growing order
    for layer_name in follow_layers_names[1 :]:
        if model.get_submodule(layer_name).weight.shape[1] < model.get_submodule(smallest_following_layer_name).weight.shape[1]:
            smallest_following_layer_name = layer_name

    channels_produced_by_target_layer_num = model.get_submodule(target_layer_name).weight.shape[0]
    smallest_following_layer = model.get_submodule(smallest_following_layer_name)
    additional_channels = smallest_following_layer.weight.shape[1] - channels_produced_by_target_layer_num
    # print(target_layer_name, ": ", [additional_channels, additional_channels + channels_produced_by_target_layer_num])
    # return [additional_channels, additional_channels + channels_produced_by_target_layer_num]

    if len(follow_layers_names) > 1: # It doesn't mean results won't be concat however it always means there is empty concat_group
        denseblock, denselayer, = target_layer_name.split('.')[1:3]
        start_denselayer_id, end_denselayer_id = int(denselayer[10:]), len(model.get_submodule(f"features.{denseblock}"))+1
        concat_group = [ f'features.{denseblock}.denselayer{layer_id}.conv2' for layer_id in range(start_denselayer_id+1, end_denselayer_id) ]
    else:
        concat_group = []
    # if denseblock == "denseblock4" and target_layer_name.split('.')[-1] == "conv2":
    #         concat_group += ["classifier.1"]
    return {
                "start": additional_channels,
                "end": additional_channels+channels_produced_by_target_layer_num,
                "concatenation_group" : concat_group
            }


def get_layer_pairs_densenet110_classic() -> LayerPairs:
    model = thh.DenseNet121TAT()
    layer_pairs = LayerPairs(name="LP | DenseNet121 | only first convlayer from each denselayer | classic")
    for denseblock_id in range(1, 5):
        denselayers_num = len( model.get_submodule(f"features.denseblock{denseblock_id}") )
        for denselayer_id in range(1, denselayers_num+1):
            layer_pairs.append({
                'target_layer' : f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                'follow_layers' : [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2'],
                'optional': [ f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.norm2' ],
                "coresponding_channels_slice" : get_channels_slice_densenet(
                                                                                model,
                                                                                f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                                                                                [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2']
                                                                            )
            })
    return layer_pairs


def get_layer_pairs_densenet110_experimental() -> LayerPairs:
    model = thh.DenseNet121TAT()
    layer_pairs = LayerPairs(name="LP | DenseNet121 | all layer from denselayer without transitions | experimental")
    for denseblock_id in range(1, 4):
        denselayers_num = len( model.get_submodule(f"features.denseblock{denseblock_id}") )
        for denselayer_id in range(1, denselayers_num+1):
            # adding first conv layer to list
            layer_pairs.append({ 
                'target_layer' : f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                'follow_layers' : [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2'],
                'optional': [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.norm2'],
                "coresponding_channels_slice" : get_channels_slice_densenet(
                                                                                 model,
                                                                                 f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                                                                                 [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2']
                                                                                )
            })
            # adding second conv layer to list
            following_layers = [ f'features.denseblock{denseblock_id}.denselayer{id}.conv1' for id in range(denselayer_id+1, denselayers_num+1) ]
            optional_layers = [ f'features.denseblock{denseblock_id}.denselayer{id}.norm1' for id in range(denselayer_id+1, denselayers_num+1) ]
            following_layers += [ f'features.transition{denseblock_id}.conv']
            optional_layers += [ f'features.transition{denseblock_id}.norm']
            target_layer = f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2'
            layer_pairs.append({
                'target_layer' : target_layer,
                'follow_layers' : following_layers,
                'optional': optional_layers,
                "coresponding_channels_slice" : get_channels_slice_densenet(
                                                                                 model,
                                                                                 target_layer,
                                                                                 following_layers
                                                                                )
            })
    for denseblock_id in range(4, 5): # !!!!!!!!!!!!!!!
        denselayers_num = len( model.get_submodule(f"features.denseblock{denseblock_id}") )
        for denselayer_id in range(1, denselayers_num+1): # why it was without last conv section? 
            # adding first conv layer to list
            layer_pairs.append({ 
                'target_layer' : f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                'follow_layers' : [ f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2' ],
                'optional': [ f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.norm2' ],
                "coresponding_channels_slice" : get_channels_slice_densenet(
                                                                                 model,
                                                                                 f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv1',
                                                                                 [f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2']
                                                                                ) 
            })

            # adding second conv layer to list
            following_layers = [ f'features.denseblock{denseblock_id}.denselayer{id}.conv1' for id in range(denselayer_id+1, denselayers_num+1) ]
            optional_layers = [ f'features.denseblock{denseblock_id}.denselayer{id}.norm1' for id in range(denselayer_id+1, denselayers_num+1) ]
            following_layers += [ f'classifier.1' ]
            optional_layers += [ f'features.norm5' ]
            target_layer = f'features.denseblock{denseblock_id}.denselayer{denselayer_id}.conv2'
            layer_pairs.append({
                'target_layer' : target_layer,
                'follow_layers' : following_layers,
                'optional': optional_layers,
                "coresponding_channels_slice" : get_channels_slice_densenet(
                                                                                 model,
                                                                                 target_layer,
                                                                                 following_layers
                                                                                ) 
            })
    return layer_pairs 


def get_hierarchical_groups_4h_densenet110_classic() -> HierarchicalGroups:
    model = thh.DenseNet121TAT()
    hierarchical_groups = HierarchicalGroups("HG | DenseNet121 | 4 hierarchies | classic | according to output size")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []

    for group in hierarchical_groups.keys():
        group_id = int(group[-1])

        denselayers_num = len( model.get_submodule(f"features.denseblock{group_id}") )
        for denselayer_id in range(1, denselayers_num+1):
            hierarchical_groups[group].append(f'features.denseblock{group_id}.denselayer{denselayer_id}.conv1')
    return hierarchical_groups


def get_hierarchical_groups_1h_densenet110_classic() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups(" HG | DenseNet121 | 1 hierarchy | classic | global pruning")
    hierarchical_groups['group1'] = {"pairs" : []}
    classic_layer_pairs = get_layer_pairs_densenet110_classic()
    for pair in classic_layer_pairs:
        hierarchical_groups['group1']['pairs'].append(pair['target_layer'])
    return hierarchical_groups


def get_hierarchical_groups_4h_densenet110_experimental() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | DenseNet121 | 4 hierarchies | experimental | according to output size")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    hierarchical_groups['group4'] = []
    experimental_layer_pairs = get_layer_pairs_densenet110_experimental()

    for group in hierarchical_groups.keys():
        group_id = int(group[-1])
        coresponding_dense_block = f"features.denseblock{group_id}"

        for pair in experimental_layer_pairs:
            if pair['target_layer'].startswith(coresponding_dense_block):
                hierarchical_groups[group].append(pair['target_layer'])
    return hierarchical_groups


def get_hierarchical_groups_1h_densenet110_experimental() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | DenseNet121 | 1 hierarchy | experimental | global pruning")
    hierarchical_groups['group1'] = []
    experimental_layer_pairs = get_layer_pairs_densenet110_experimental()
    for pair in experimental_layer_pairs:
        hierarchical_groups['group1'].append(pair['target_layer'])
    return hierarchical_groups



######################################################################### GOOGLENET #####################################################
inception_modules_in_order = [
    "inception3a",
    "inception3b",
    "inception4a",
    "inception4b",
    "inception4c",
    "inception4d",
    "inception4e",
    "inception5a",
    "inception5b",
]

inception_modules_as_3h = {
    "group1" : ["inception3a", "inception3b"],
    "group2" : ["inception4a", "inception4b", "inception4c", "inception4d", "inception4e"],
    "group3" : ["inception5a", "inception5b"]
}

branch_last_layers = {
    "branch1" : "conv",
    "branch2" : "1.conv",
    "branch3" : "1.conv",
    "branch4" : "1.conv",
}

branch_connection_layers = {
    "branch1" : "conv",
    "branch2" : "0.conv",
    "branch3" : "0.conv",
    "branch4" : "1.conv",
}

def get_layer_pairs_googlenet_classic_v1_not_to_use() -> LayerPairs:
    model = thh.GoogLeNetTAT()
    layer_pairs = LayerPairs(name="LP | write desc")

    for inception_name in inception_modules_in_order:
        for branch_id in range(2, 4):
            layer_pairs.append({
                "target_layer" : f"{inception_name}.branch{branch_id}.0.conv",
                "follow_layers" : [f"{inception_name}.branch{branch_id}.1.conv"],
                "optional" : [f"{inception_name}.branch{branch_id}.0.bn"],
                "coresponding_channels_slice" : get_channels_slice_googlenet(
                                                                                 model,
                                                                                 f"{inception_name}.branch{branch_id}.0.conv",
                                                                                 [f"{inception_name}.branch{branch_id}.1.conv"]
                                                                                ) 
            })
    return layer_pairs


def get_layer_pairs_googlenet_classic_v2() -> LayerPairs:
    model = thh.GoogLeNetTAT()
    layer_pairs = LayerPairs(name="LP | write desc")
    layer_pairs_cl = get_layer_pairs_googlenet_classic_v1_not_to_use()

    layer_pairs.append({        
        "target_layer" : f"conv2.conv",
        "follow_layers" : [f"conv3.conv"],
        "optional" : [f"conv2.bn"],
        "coresponding_channels_slice" : get_channels_slice_googlenet(
                                                                        model,
                                                                        "conv2.conv",
                                                                        [f"conv3.conv"]
                                                                    )     
    })
    for pair in layer_pairs_cl:
        layer_pairs.append(pair)
    return layer_pairs


def get_layer_pairs_googlenet_experimental() -> LayerPairs:
    model = thh.GoogLeNetTAT()
    layer_pairs = LayerPairs(name="LP | GoogLeNet | all layers from all inception modules | experimental")
    # adding 2nd conv layer before inceptions
    layer_pairs.append({        
        "target_layer" : f"conv2.conv",
        "follow_layers" : [f"conv3.conv"],
        "optional" : [f"conv2.bn"],
        "coresponding_channels_slice" : get_channels_slice_googlenet(
                                                                        model,
                                                                        "conv2.conv",
                                                                        [f"conv3.conv"]
                                                                    )     
    })
    # adding 3rd conv layer before inceptions
    f_layers = [
                'inception3a.branch1.conv',
                'inception3a.branch2.0.conv',
                'inception3a.branch3.0.conv',
                'inception3a.branch4.1.conv'
                ]
    layer_pairs.append({        
        "target_layer" : f"conv3.conv",
        "follow_layers" : f_layers,
        "optional" : [f"conv3.bn"],
        "coresponding_channels_slice" : {
            "start": 0,
            "end": model.get_submodule(f"conv3.conv").weight.shape[0],
            "concatenation_group" : []
        }
    })
    
    for inc_id, inception_name in enumerate(inception_modules_in_order):
        for branch_id in range(2, 4): # classic layers from inception
            layer_pairs.append({
                "target_layer" : f"{inception_name}.branch{branch_id}.0.conv",
                "follow_layers" : [f"{inception_name}.branch{branch_id}.1.conv"],
                "optional" : [f"{inception_name}.branch{branch_id}.0.bn"],
                "coresponding_channels_slice" : get_channels_slice_googlenet(
                                                                                 model,
                                                                                 f"{inception_name}.branch{branch_id}.0.conv",
                                                                                 [f"{inception_name}.branch{branch_id}.1.conv"]
                                                                                ) 
            })
        for branch_id in range(1, 5): # last layers from inception 
            branch = f"branch{branch_id}"
            next_inception_name = inception_modules_in_order[inc_id+1] if inc_id != len(inception_modules_in_order)-1 else None
            target_layer = f"{inception_name}.{branch}.{branch_last_layers[branch]}"
            follow_layers = [ f"{next_inception_name}.branch{id}.{branch_connection_layers[ 'branch' + str(id)]}" for id in range(1, 5) ] if inception_name!="inception5b" else ["fc.0"]
            optional_layers = [ f"{inception_name}.{branch}.1.bn"] if branch_id != 1 else [ f"{inception_name}.{branch}.bn"]
            # for inception4a and inception4d there is additional aux1 and aux2 connected.
            if inception_name=="inception4a":
                follow_layers += [ "aux1.conv.conv" ]
            elif inception_name=="inception4d":
                follow_layers += [ "aux2.conv.conv"]

            layer_pairs.append({
                "target_layer" : target_layer,
                "follow_layers" : follow_layers,
                "optional" : optional_layers,
                "coresponding_channels_slice" : get_channels_slice_googlenet(
                                                                                 model,
                                                                                 target_layer,
                                                                                 follow_layers
                                                                                ) 
            })
    return layer_pairs    


def get_hierarchical_groups_3h_googlenet_experimental() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | GoogLeNet | 3 hierarchies | experimental | according to output size")
    
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []
    
    for group in hierarchical_groups.keys():
        for inc_id, inception_name in enumerate(inception_modules_as_3h[group]):
            for branch_id in range(2, 4): # classic layers from inception
                hierarchical_groups[group].append(f"{inception_name}.branch{branch_id}.0.conv")
                  
            for branch_id in range(1, 5): # last layers from inception 
                branch = f"branch{branch_id}"
                target_layer = f"{inception_name}.{branch}.{branch_last_layers[branch]}"
                hierarchical_groups[group].append(target_layer)
    
    # first two layers
    hierarchical_groups['group1'] += ["conv2.conv", "conv3.conv"]
    

    return hierarchical_groups


def get_hierarchical_groups_1h_googlenet_experimental() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | GoogLeNet | 1 hierarchy | experimental | global pruning")
    hierarchical_groups['group1'] = []
    experimental_layer_pairs = get_layer_pairs_googlenet_experimental()
    for pair in experimental_layer_pairs:
        hierarchical_groups['group1'].append( pair['target_layer'] )
    return hierarchical_groups


def get_hierarchical_groups_3h_googlenet_classic() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | GoogLeNet | 3 hierarchies | classic | according to output size")
    hierarchical_groups['group1'] = []
    hierarchical_groups['group2'] = []
    hierarchical_groups['group3'] = []

    for group in hierarchical_groups.keys():
        for inc_id, inception_name in enumerate(inception_modules_as_3h[group]):
            for branch_id in range(2, 4): # classic layers from inception
                hierarchical_groups[group].append(f"{inception_name}.branch{branch_id}.0.conv")
    hierarchical_groups['group1'].append(f"conv2.conv")
    return hierarchical_groups


def get_hierarchical_groups_1h_googlenet_classic() -> HierarchicalGroups:
    hierarchical_groups = HierarchicalGroups("HG | GoogLeNet | 1 hierarchy | classic | global pruning")
    hierarchical_groups['group1'] = []
    classic_layer_pairs = get_layer_pairs_googlenet_classic_v2()
    for pair in classic_layer_pairs:
        hierarchical_groups['group1'].append(pair['target_layer'])
    return hierarchical_groups


def get_channels_slice_googlenet(model: th.nn.Module, target_layer_name: str, follow_layers_names: list[str]) -> dict:
    target_inception = target_layer_name.split('.')[0] if target_layer_name.split('.')[0] in inception_modules_in_order else None
    follow_inception = follow_layers_names[0].split('.')[0] if follow_layers_names[0].split('.')[0] in inception_modules_in_order else None

    channels_produced_by_target_layer_num = model.get_submodule(target_layer_name).weight.shape[0]
    if target_inception == None or target_inception == follow_inception: # this means all channels from follow layer were produced by target layer. (Not depthconcat involved)
        additional_channels = 0
        return {
                "start": additional_channels,
                "end": additional_channels+channels_produced_by_target_layer_num,
                "concatenation_group" : []
                }
    else:
        additional_channels = 0
        current_branch = "branch1"
        target_branch = target_layer_name.split('.')[1]
        while current_branch != target_branch:
            additional_channels += model.get_submodule(f"{target_inception}.{current_branch}.{branch_last_layers[current_branch]}").weight.shape[0]
            current_branch = current_branch[:-1] + f"{int(current_branch[-1]) + 1}"

        # all branches after target_branch
        concat_group = [f"{target_inception}.{f'branch{branch_id}'}.{branch_last_layers[f'branch{branch_id}']}" for branch_id in range(int(target_branch[-1:])+1, 5)]
            
        return {
                "start": additional_channels,
                "end": additional_channels+channels_produced_by_target_layer_num,
                "concatenation_group" : concat_group
                }


if __name__ == '__main__':
    pass
