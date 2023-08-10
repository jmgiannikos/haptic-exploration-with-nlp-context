import random
import clip
import torch
import numpy as np

import grasp_cls_pipeline_configs as configs

def split_nlp_prompt_dict(folds):
    prompts = np.load(configs.get_language_prompts_path())
    train_test_dicts = [{"train":{},"val":{}}]*folds
    for key in prompts.keys():
        prompt_list = prompts[key]
        block_size = int(len(prompt_list)/folds)
        for fold in range(folds):
            if fold != folds-1:
                start_idx = block_size*fold
                end_idx = start_idx+block_size
            else:
                start_idx = block_size*fold
                end_idx = -1

            train_test_dicts[fold]["val"][key] = prompt_list[start_idx:end_idx]
            train_test_dicts[fold]["train"][key] = [prompt for prompt in prompt_list if prompt not in train_test_dicts[fold]["val"][key]]

    return train_test_dicts

def get_language_prompt(object_types, prompts):
    if not isinstance(object_types, tuple):
        object_types = [object_types]

    selected_prompts = []
    for object_type in object_types:
        prompt_list = prompts[object_type]
        prompt = random.choice(prompt_list)
        selected_prompts.append(prompt)
    return selected_prompts

def get_language_embeddings(object_types, object_type_dict, nlp_promts, clip_nlp_model):
    if not isinstance(object_types, tuple):
        object_types = [object_types]

    if not configs.get_calculate_language_embedding():
        for obj_type_idx, object_type in enumerate(object_types):
            if obj_type_idx == 0:
                language_embedding = object_type_dict[object_type][None,:]
            else:
                language_embedding = torch.cat((language_embedding, object_type_dict[object_type][None,:]), 0)
        language_embedding = language_embedding.float()
    else:
        language_embedding = get_clip_language_embedding(nlp_prompts=nlp_promts, object_type=object_types, clip_model=clip_nlp_model)

    return language_embedding 

def get_clip_language_embedding(nlp_prompts, object_type, clip_model):
    nlp_promt = random.choice(nlp_prompts[object_type])
    prompt_tokens = clip.tokenize(nlp_promt).to(configs.get_device())
    prompt_embedding = clip_model.encode_text(prompt_tokens)
    return prompt_embedding

def generate_object_type_dict(dataset_paths):
    type_dict = {}
    index = 0
    for name in dataset_paths:
        object_type = name.split("/")[-1].split("_")[1]
        if object_type not in type_dict.keys():
            type_dict[object_type] = index
            index += 1

    for key in type_dict:
        idx_list = [0]*len(list(type_dict.keys()))
        idx_list[type_dict[key]] = 1
        idx_tensor = torch.tensor(idx_list).to(configs.get_device())
        type_dict[key] = idx_tensor

    return type_dict