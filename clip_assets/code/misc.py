import random
import numpy as np
import torch
from datetime import datetime



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
#    if args.n_gpu > 0:
#        torch.cuda.manual_seed_all(args.seed)

def print_info(content):
    print("Time: {}\t{}\n".format(datetime.now().strftime("%H:%M:%S"), content)) 