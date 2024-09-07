import os.path
import torch
from utils import get_intent_labels, get_slot_labels, MODEL_CLASSES


def load_args(model_name, json_file_path):
    parser = torch.load(os.path.join(model_name, json_file_path))
    return parser


def load_model(model_dir, args, device):
    model = MODEL_CLASSES[args.model_type][1].from_pretrained(model_dir,
                                                              args=args,
                                                              intent_label_lst=get_intent_labels(args),
                                                              slot_label_lst=get_slot_labels(args))
    model.to(device)
    model.eval()