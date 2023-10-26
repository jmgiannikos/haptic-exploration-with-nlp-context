import os
import numpy as np
import grasp_cls_utils as utils
import json

NLP_TRAIN_PROMPT_FILE_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/gpt_generated_object_prompt_dataset.npz"
NLP_PROMPT_JSON = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/grasp_classification/gpt_generated_object_prompt_dataset.json"
NLP_USE_PROMPT_FILE_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/usable_gpt_generated_object_prompt_dataset.npz"

def make_both_datasets(json_path, train_path, use_path):
    f = open(json_path,mode="r")
    new_data_dict = json.load(f)
    make_use_dataset(new_data_dict=new_data_dict, use_path=use_path)
    make_train_dataset(new_data_dict=new_data_dict, train_path=train_path)
    

def make_train_dataset(new_data_dict, train_path, add_to_file=False):
    data_dict = {"nlp prompt":[], "object type":[]}
    for key in new_data_dict.keys():
        for prompt in new_data_dict[key]:
            data_dict["nlp prompt"].append(prompt)
            data_dict["object type"].append(key)

    if os.path.isfile(train_path) and add_to_file:
        add_dict = np.load(train_path)
        for key in data_dict.keys():
            utils.dict_list_append(key, data_dict[key], add_dict)
    else:
        add_dict = data_dict

    np.savez(train_path, **add_dict)

def make_use_dataset(use_path, new_data_dict, add_to_file=False):
    if os.path.isfile(use_path) and add_to_file:
        add_use_dict = np.load(use_path)
        for key in new_data_dict.keys():
            utils.dict_list_append(key, new_data_dict[key], add_use_dict)
    else:
        add_use_dict = new_data_dict

    np.savez(use_path, **add_use_dict)

def main():
    make_both_datasets(json_path=NLP_PROMPT_JSON, use_path=NLP_USE_PROMPT_FILE_PATH, train_path=NLP_TRAIN_PROMPT_FILE_PATH)


if __name__ == '__main__':
    main()