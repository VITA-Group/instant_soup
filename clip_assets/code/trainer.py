import os
import time
import torch
import random
import copy
from misc import print_info
from pruner import Pruner
from modeling import ImageEncoder, ImageClassifier
from heads import get_classification_head
from datasets.registry import get_dataset
from utils import cosine_lr, LabelSmoothing
from eval import eval_single_dataset, evaluate
from datasets.common import get_dataloader, maybe_dictionarize
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

class Trainer(object):
    def __init__(self, args):
        for k, v in vars(args).items():
            setattr(self, k, v)
        self.args = args
        
        self.image_encoder = ImageEncoder(args, keep_lang=False)
        self.classification_head = get_classification_head(args, args.train_dataset)

        self.model = ImageClassifier(self.image_encoder, self.classification_head)
        self.model.freeze_head()

        preprocess_fn = self.model.train_preprocess
        self.print_every = 100
        self.dataset = get_dataset(
            args.train_dataset,
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        num_batches = len(self.dataset.train_loader)
        self.data_loader = get_dataloader(
            self.dataset, is_train=True, args=args, image_encoder=None)

        devices = list(range(torch.cuda.device_count()))
        print('Using devices', devices)
        # self.model = torch.nn.DataParallel(self.model, device_ids=devices)
        self.model = self.model.cuda()

        if args.ls > 0:
            self.loss_fn = LabelSmoothing(args.ls)
        else:
            self.loss_fn = torch.nn.CrossEntropyLoss()

        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(self.params, lr = self.lr, weight_decay = args.wd)
        self.t_total = args.epochs * num_batches
        self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=args.warmup_length, num_training_steps=args.epochs * num_batches
        )
        # print("-*-*-*-*-*-*-*-*-*-*-*- Trainer Statistics -*-*-*-*-*-*-*-*-*-*-*-")
        # print_info("Task Name    = {}".format(len(self.task_name)))
        # print_info("Num Examples = {}".format(len(self.train_dataset)))
        # print_info("Num Epochs   = {}".format(self.num_train_epochs))
        # print_info("Total optimization steps    = {}".format(self.t_total))
        # print("-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-")

        self.global_step, self.epoch_trained = 0, 0
        self.training_loss = [-1.0]
        self.evaluation_result = {}

        #pruning related unilities 
        self.pruner = Pruner(self.model.image_encoder)
        print('Image Encoder Done LR !!')

        

    def save_model(self, prefix = ""):
        output_dir = os.path.join(".", "checkpoints")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(self.model, os.path.join(output_dir, "{}_model_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio())))
        # torch.save(self.optimizer, os.path.join(output_dir, "optimizer.pt"))
        torch.save(self.pruner.get_prune_mask(), os.path.join(output_dir, "{}_mask_{}_{}.pt".format(prefix, self.args.train_dataset, self.pruner.get_sparsity_ratio())))
        
        
    def evaluate_model(self):
        results = eval_single_dataset(self.image_encoder, self.args.train_dataset, self.args)
        return results

    def train_epoch(self, prob = 1.0, seed = 99):
        print_info("Training Epoch => {} || Learning Rate => {} || Current Loss => {:.3f}".format(self.epoch_trained + 1, 
                                                            self.scheduler._last_lr, self.training_loss[self.epoch_trained]))
        
        epoch_loss = 0.0
        random.seed(seed)
        self.model.train()

        for i, batch in enumerate(self.data_loader):
            if random.random() > prob:
                continue
            start_time = time.time()
            self.optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].cuda()
            labels = batch['labels'].cuda()
            data_time = time.time() - start_time

            logits = self.model(inputs)
            loss = self.loss_fn(logits, labels)
            loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_norm_(self.params, 1.0)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            batch_time = time.time() - start_time
            self.global_step += 1

            if i % self.print_every == 0:
                percent_complete = 100 * i / len(self.data_loader)
                print(
                    f"Train Epoch: {self.epoch_trained + 1} [{percent_complete:.0f}% {i}/{len(self.dataset.train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
        
        self.epoch_trained = self.epoch_trained + 1
        self.training_loss.append(loss.item())

    def get_state_dict(self):
        return self.model.image_encoder.state_dict(), self.optimizer.state_dict(), [self.scheduler._last_lr, self.scheduler.last_epoch]

    def set_state_dict(self, model_state, optimizer_state = None, scheduler_state = None):
        self.model.image_encoder.load_state_dict(model_state)
        if optimizer_state != None:
            self.optimizer = torch.optim.AdamW(self.params, lr = scheduler_state[0][0], weight_decay = self.args.wd)
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=self.args.warmup_length, num_training_steps=1000
            )
            # self.optimizer.load_state_dict(optimizer_state)
            # self.scheduler =  get_linear_schedule_with_warmup(
            #     self.optimizer, num_warmup_steps=self.warmup_length, num_training_steps = self.t_total - scheduler_state[1]
            # )
            # self.scheduler.base_lrs = scheduler_state[0]
        print("Model and Optimizer State re-initialized.")
