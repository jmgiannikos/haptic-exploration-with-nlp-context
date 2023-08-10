import numpy as np
from torch.utils.data import Dataset
import os
import json

import nlp_cls_pipeline_configs as configs
import grasp_cls_utils as gcls_utils

def object_type_to_class_idx(labels):
    if os.path.isfile(configs.get_object_type_json_path()):
        f = open(configs.get_object_type_json_path(),mode="r")
        label_dict = json.load(f)
    else:
        label_dict = {}
    result_list = []
    max_idx = len(label_dict)-1
    for label in labels:
        if label in label_dict.keys():
            result_list.append(label_dict[label])
        else:
            max_idx += 1
            label_dict[label] = max_idx
            result_list.append(max_idx)
    s = open(configs.get_object_type_json_path(),mode="w")
    json.dump(label_dict,s)
    return result_list

def class_idxs_to_one_hot(class_idxs):
    num_idxs = len(set(class_idxs))
    result_array = None
    for class_idx in class_idxs:
        one_hot = [0]*num_idxs
        one_hot[class_idx] = 1
        result_array = gcls_utils.array_append(one_hot, result_array, 0)
    return result_array

class nlp_dataset(Dataset):
    def __init__(self, data=None, labels=None):
        self.labels = class_idxs_to_one_hot(object_type_to_class_idx(labels))
        self.nlp_prompts = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        nlp_prompt = self.nlp_prompts[idx]
        label = self.labels[idx]

        return nlp_prompt, label