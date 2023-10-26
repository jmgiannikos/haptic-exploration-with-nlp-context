import os
import numpy as np

NLP_PROMPT_FILE_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/grasp_classification/complex_objects_nlp_prompts.npz"
ADD_DICT = {
    "mircrophone1": [
        "microphone",
        "mic",
        "audio recorder",
        "handheld microphone",
        "voice recorder"
    ],
    "rubber": [
        "rubber duck",
        "duck",
        "animal toy",
        "ducky",
        "squeaky toy"
    ],
    "hair": [
        "hair dryer",
        "blow-dryer",
        "hand blower",
        "blow drier",
        "hair drier"
    ],
    "can1": [
        "can",
        "soda can",
        "aluminium can",
        "cylindrical object",
        "canister"
    ],
    "camera": [
        "camera",
        "digital camera",
        "photo camera",
        "handheld camera",
        "cam"
    ],
    "bottle": [
        "bottle",
        "glass bottle",
        "water bottle",
        "wine",
        "flask"
    ]
}
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