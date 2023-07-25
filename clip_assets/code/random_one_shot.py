import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
import torch
import copy
from trainer import Trainer
from args import parse_arguments

def main():
    args = parse_arguments()

    fopen = open("logs/random_Cars_MNIST_SVHN_GTSRB.txt", "a")
    main_trainer = Trainer(args)
    print("CLIP Trainer Created ...")

    target_sparsity = args.target_sparsity / 100

    main_trainer.pruner.prune_model(target_sparsity, isRandom=True)

    max_log = []
    fopen.write(f"------------ Target Sparsity : {args.target_sparsity} || Dataset : {args.train_dataset}-------------\n")
    for i in range(0, 15):
        main_trainer.train_epoch(1.0, args.seed)
        res = main_trainer.evaluate_model()
        fopen.write("Main Trainer {} sparsity : {:.4} % and performance : {}\n".format(i, main_trainer.pruner.get_sparsity_ratio(), res))
        fopen.flush()
        max_log.append(res["top1"])

    fopen.write("{}".format(max_log))
    fopen.write(f"\nSparsity: {main_trainer.pruner.get_sparsity_ratio()} \t Result: {max(max_log)}\n\n")
    fopen.close()

if __name__ == "__main__":
    main()