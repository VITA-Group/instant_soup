import os
from re import L
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
import numpy as np
import torch
import random
import copy
from trainer import Trainer
from args import parse_arguments

import os


def load_mask_array(mask):
    total = 0
    for key in mask:
        total += mask[key].numel()
    mask_array = torch.zeros(total)
    index = 0
    for key in mask:
        size = mask[key].numel()
        mask_array[index:(index + size)] = mask[key].view(-1).abs().clone()
        index += size
    return mask_array

def main():
    files = sorted(os.listdir("/data/sparse_soup/lth_cifar10"))
    print(files)    
    fopen = open("logs/sim_score.txt", "w")
    args = parse_arguments()
    cos = torch.nn.CosineSimilarity(dim=0)
    prune_ratio = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
    i = 0
    for file in files:
        print(f"-----------------------{file}----------------------")
        mask_tensor = load_mask_array(torch.load("/data/sparse_soup/lth_cifar10/"+files[1]))

        main_trainer = Trainer(args)
        main_trainer.pruner.prune_model(prune_ratio[i]/100.0)
        print("Main Trainer sparsity == {} %".format(main_trainer.pruner.get_sparsity_ratio()))
        lth_mask = main_trainer.pruner.get_prune_mask()
        lth_tensor = load_mask_array(lth_mask)
        print("Cosine similarity: {}".format(cos(mask_tensor, lth_tensor)))
        fopen.write(f"Prune Ratio: {prune_ratio[i]} \t Similairty: {cos(mask_tensor, lth_tensor)}\n")
        fopen.flush()
        i += 1
    fopen.close()


if __name__ == "__main__":
    main()