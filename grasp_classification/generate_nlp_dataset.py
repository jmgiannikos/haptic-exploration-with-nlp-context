import os
import numpy as np
import grasp_cls_utils as utils
import json

NLP_TRAIN_PROMPT_FILE_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_dataset.npz"
NLP_PROMPT_JSON = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/new_nlp_prompts.json"
NLP_USE_PROMPT_FILE_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_use_dataset.npz"

def main():
    f = open(NLP_PROMPT_JSON,mode="r")
    new_data_dict = json.load(f)
    data_dict = {"nlp prompt":[], "object type":[]}

    for key in new_data_dict.keys():
        for prompt in new_data_dict[key]:
            data_dict["nlp prompt"].append(prompt)
            data_dict["object type"].append(key)

    if os.path.isfile(NLP_TRAIN_PROMPT_FILE_PATH):
        add_dict = np.load(NLP_TRAIN_PROMPT_FILE_PATH)
        for key in data_dict.keys():
            utils.dict_list_append(key, data_dict[key], add_dict)
    else:
        add_dict = data_dict

    if os.path.isfile(NLP_USE_PROMPT_FILE_PATH):
        add_use_dict = np.load(NLP_USE_PROMPT_FILE_PATH)
        for key in new_data_dict.keys():
            utils.dict_list_append(key, new_data_dict[key], add_use_dict)
    else:
        add_use_dict = new_data_dict

    np.savez(NLP_TRAIN_PROMPT_FILE_PATH, **add_dict)
    np.savez(NLP_USE_PROMPT_FILE_PATH, **add_use_dict)


if __name__ == '__main__':
    main()