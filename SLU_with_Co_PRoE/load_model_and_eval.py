import argparse
import os.path
import torch
from data_loader import load_and_cache_examples
from datetime import datetime
import random
import time
import json
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from trainer import Trainer_multi, Trainer
from utils import init_logger, load_tokenizer, get_intent_labels, get_slot_labels, MODEL_CLASSES, set_seed


def load_args(model_name, json_file_path):
    parser = torch.load(os.path.join(model_name, json_file_path))
    return parser

def load_model(model_dir, args, device):
    # Check whether model exists
    # if not os.path.exists(model_dir):
    #     raise Exception("Model doesn't exists! Train first!")

    # try:
    model = MODEL_CLASSES[args.model_type][1].from_pretrained(model_dir,
                                                              args=args,
                                                              intent_label_lst=get_intent_labels(args),
                                                              slot_label_lst=get_slot_labels(args))
    model.to(device)
    model.eval()
    # except:
    #     raise Exception("Some model files might be missing...")
def dodo(args):
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    if args.multi_intent == 1:
        trainer = Trainer_multi(args, train_dataset, dev_dataset, test_dataset)
    else:
        trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)

    if args.do_train:
        print(trainer.evaluate('test'))

    # return model
mixatis_path = os.path.join('mixatis_model', 'pytorch_model.bin')
mixsnips_path = os.path.join('mixsnips', 'pytorch_model.bin')

args_mixatis = load_args('mixatis_model', 'training_args.bin')
args_mixsnips = load_args('mixsnips_model', 'training_args.bin')

args_mixatis.model_dir = 'mixatis_model'
args_mixsnips.model_dir = 'mixsnips_model'

load_model(args_mixatis.model_dir, args_mixatis, 'cuda')
load_model(args_mixsnips.model_dir, args_mixsnips, 'cuda')

dodo(args_mixatis)
dodo(args_mixsnips)
