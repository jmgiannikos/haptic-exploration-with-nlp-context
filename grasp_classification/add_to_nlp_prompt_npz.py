import os
import numpy as np

NLP_PROMPT_FILE_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/simple_nlp_prompts.npz"
ADD_DICT = {"ico":["ico sphere"],
            "cylinder":["lying cylinder"],
            "block":["block"],
            "cube":["cube"],
            "cylinder2":["upright cylinder"]}

def dict_list_append(key, val, target_dict):
    if key in target_dict.keys():
        target_dict[key].append(val)
    else:
        target_dict[key] = [val]

def main():
    if os.path.isfile(NLP_PROMPT_FILE_PATH):
        add_dict = np.load(NLP_PROMPT_FILE_PATH)
        for key in ADD_DICT.keys():
            dict_list_append(key, ADD_DICT[key], add_dict)

    else:
        add_dict = ADD_DICT

    np.savez(NLP_PROMPT_FILE_PATH, **add_dict)


if __name__ == '__main__':
    main()