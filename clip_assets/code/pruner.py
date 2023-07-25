import copy
from unittest import makeSuite
import torch
import torch.nn.utils.prune as prune

class Pruner(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.mask_parameters = []
        self.prune_module_name = []
        self.isPruned = False
        for ii in range(12):
            self.mask_parameters.append((self.encoder.model.visual.transformer.resblocks[ii].attn, 'in_proj_weight'))
            self.prune_module_name.append('model.visual.transformer.resblocks.{}.attn'.format(ii))

            self.mask_parameters.append((self.encoder.model.visual.transformer.resblocks[ii].attn.out_proj, 'weight'))
            self.prune_module_name.append('model.visual.transformer.resblocks.{}.attn.out_proj'.format(ii))

            self.mask_parameters.append((self.encoder.model.visual.transformer.resblocks[ii].mlp.c_fc, 'weight'))
            self.prune_module_name.append('model.visual.transformer.resblocks.{}.mlp.c_fc'.format(ii))

            self.mask_parameters.append((self.encoder.model.visual.transformer.resblocks[ii].mlp.c_proj, 'weight'))
            self.prune_module_name.append('model.visual.transformer.resblocks.{}.mlp.c_proj'.format(ii))

        print("Pruner Initialized")

    def get_sparsity_ratio(self):
        print("Is Model Pruned : {}".format(self.isPruned))
        sum_list = 0
        zero_sum = 0
        for module, _ in self.mask_parameters:
            if isinstance(module, torch.nn.MultiheadAttention):
                sum_list += float(module.in_proj_weight.nelement())
                zero_sum += float(torch.sum(module.in_proj_weight == 0))
                continue
            sum_list += float(module.weight.nelement())
            zero_sum += float(torch.sum(module.weight == 0))
        return 100*zero_sum/sum_list

    def prune_model(self, per_zero, isRandom = False):
        if isRandom == False:
            prune.global_unstructured(
                tuple(self.mask_parameters),
                pruning_method=prune.L1Unstructured,
                amount=per_zero,
            )
        else:
            print("Random pruning Model")
            prune.global_unstructured(
                tuple(self.mask_parameters),
                pruning_method=prune.RandomUnstructured,
                amount=per_zero,
            )
        self.isPruned = True

    def prune_model_custom(self, mask_dict):
        print("Pruning with custom mask!")
        for name, m in self.encoder.named_modules(): 
            if name in self.prune_module_name:
                if "attn" in name:
                    if "out_proj" in name:
                        prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name])
                    else:
                        prune.CustomFromMask.apply(m, 'in_proj_weight', mask=mask_dict[name])
                else:
                    prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name])
        self.isPruned = True

    def get_prune_mask(self):
        assert self.isPruned == True, "Model is not pruned!"
        mask_dict = {}
        for name, m in self.encoder.named_modules(): 
            if name in self.prune_module_name:
                if "attn" in name:
                    if "out_proj" in name:
                        mask_dict[name] = copy.deepcopy(m.weight_mask)
                    else:
                        mask_dict[name] = copy.deepcopy(m.in_proj_weight_mask)
                else:
                    mask_dict[name] = copy.deepcopy(m.weight_mask)
        return mask_dict

    def remove_prune(self):
        assert self.isPruned == True, "Model is not pruned!"
        for name, m in self.encoder.named_modules(): 
            if name in self.prune_module_name:
                if "attn" in name:
                    if "out_proj" in name:
                        prune.remove(m, "weight")
                    else:
                        prune.remove(m, "in_proj_weight")
                else:
                    prune.remove(m, "weight")
        self.isPruned = False