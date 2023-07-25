import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import numpy as np
import torch
import copy
from trainer import Trainer
from args import parse_arguments

def main():
    args = parse_arguments()

    fopen = open("logs/Interval_{}.txt".format(args.train_dataset), "a")
    main_trainer = Trainer(args)
    main_trainer.pruner.prune_model(0.0)
    print("CLIP Trainer Created ...")

    
    epoch_count = args.epochs
    

    for i in range(0, 11):
        main_trainer.train_epoch(1.0, 99)
        main_trainer.train_epoch(1.0, 99)
        fopen.write("Interval Trainer {} sparsity : {:.4} % and performance : {}\n".format(i, main_trainer.pruner.get_sparsity_ratio(),main_trainer.evaluate_model()))
        fopen.flush()
        current_sparsity = main_trainer.pruner.get_sparsity_ratio()
        prune_ratio = 10 / (100 - current_sparsity)
        main_trainer.pruner.prune_model(prune_ratio)

if __name__ == "__main__":
    main()