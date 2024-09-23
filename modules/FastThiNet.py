import time
import sys

sys.path.append('/net/people/plgrid/plgkogel/mainproject/modules/')
from ThiNet import ThiNet


class FastThiNet(ThiNet):
    """ This class implements algorithm FastThiNet proposed in paper [].
    As it uses same mechanisms as ThiNet it just overrides function for T subset selection.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def find_subset_T(self, pair: dict, number: int=None) -> tuple:
        target_layer_name = pair['target_layer']
        num_channels = self.model.get_submodule(target_layer_name).weight.shape[0]
        I, T = [i for i in range(num_channels)], []

        # number of kernels to remove can be explicitly passed as a argument during sensitivity analysis
        kernels_to_remove_num = -1
        if number != None:
            kernels_to_remove_num = number
        elif number == None:
            kernels_to_remove_num = int(round(num_channels * self.ratio * self.additional_ratios_mask[target_layer_name], 0))
            print(f"final ratio for layer: {self.ratio * self.additional_ratios_mask[target_layer_name]} => kernels {kernels_to_remove_num}")
        
        values = dict()
        for channel in I:
            start = time.time()
            I.remove(channel)
            value = self.channels_score(I)
            I.append(channel)
            values[channel] = value

        max_value = max(values.values())

        while len(T) < kernels_to_remove_num:
            k = list( values.keys() )
            v = list( values.values() )
            min_i = k[v.index(min(v))]
            T.append( min_i )
            values[ min_i ] = max_value + 1
        return T, I
