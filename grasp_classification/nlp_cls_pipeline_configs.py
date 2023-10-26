import torch.nn as nn
import torch
import json

CLIP_MODEL_NAME = "ViT-B/32"
OBJECT_TYPE_JSON_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/object_types.json"
LOSS_CRITERION = nn.CrossEntropyLoss()
NUM_FOLDS = 5
PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/"
DATA_FILENAME = "gpt_generated_object_prompt_dataset.npz"
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NO_CROSSVAL = False
LEARNING_RATE = 0.0002
GAMMA = 0.99
RANDOM_SEED = 42
MODEL_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/nlp_cls_results_run_09.07.23_19:27:56/nlp cls complex obj_best_model_fold_0"
CROSSVAL_FOLD_LOADED = 0
CALCULATE_COSINE_SIMILARITY_MATRIX = False
FROZEN_CLIP = False

CONF_DICT = {}

def custom_json_serializer(dictionary):
    for key in dictionary.keys():
        if key == "LOSS_CRITERION":
            if isinstance(dictionary[key], nn.CrossEntropyLoss):
                dictionary[key] = "CrossEntropyLoss"
    return dictionary

def custom_json_deserializer(dictionary):
    for key in dictionary.keys():
        if key == "LOSS_CRITERION":
            if dictionary[key] == "CrossEntropyLoss":
                dictionary[key] = nn.CrossEntropyLoss()
    return dictionary


def get_frozen_clip():
    if "FROZEN_CLIP" in CONF_DICT.keys():
        return CONF_DICT["FROZEN_CLIP"]
    else:
        return FROZEN_CLIP

def get_calculate_cosine_similarity_matrix():
    if "CALCULATE_COSINE_SIMILARITY_MATRIX" in CONF_DICT.keys():
        return CONF_DICT["CALCULATE_COSINE_SIMILARITY_MATRIX"]
    else:
        return CALCULATE_COSINE_SIMILARITY_MATRIX

def get_crossval_fold_loaded():
    if "CROSSVAL_FOLD_LOADED" in CONF_DICT.keys():
        return CONF_DICT["CROSSVAL_FOLD_LOADED"]
    else:
        return CROSSVAL_FOLD_LOADED

def get_model_path():
    if "MODEL_PATH" in CONF_DICT.keys():
        return CONF_DICT["MODEL_PATH"]
    else:
        return MODEL_PATH

def get_clip_model_name():
    if "CLIP_MODEL_NAME" in CONF_DICT.keys():
        return CONF_DICT["CLIP_MODEL_NAME"]
    else:
        return CLIP_MODEL_NAME
    
def get_object_type_json_path():
    if "OBJECT_TYPE_JSON_PATH" in CONF_DICT.keys():
        return CONF_DICT["OBJECT_TYPE_JSON_PATH"]
    else:
        return OBJECT_TYPE_JSON_PATH
    
def get_loss_criterion():
    if "LOSS_CRITERION" in CONF_DICT.keys():
        return CONF_DICT["LOSS_CRITERION"]
    else:
        return LOSS_CRITERION
    
def get_num_folds():
    if "NUM_FOLDS" in CONF_DICT.keys():
        return CONF_DICT["NUM_FOLDS"]
    else:
        return NUM_FOLDS

def get_num_epochs():
    if "NUM_EPOCHS" in CONF_DICT.keys():
        return CONF_DICT["NUM_EPOCHS"]
    else:
        return NUM_EPOCHS

def get_device():
    if "DEVICE" in CONF_DICT.keys():
        return CONF_DICT["DEVICE"]
    else:
        return DEVICE
    
def get_no_crossval():
    if "NO_CROSSVAL" in CONF_DICT.keys():
        return CONF_DICT["NO_CROSSVAL"]
    else:
        return NO_CROSSVAL
    
def get_learning_rate():
    if "LEARNING_RATE" in CONF_DICT.keys():
        return CONF_DICT["LEARNING_RATE"]
    else:
        return LEARNING_RATE
    
def get_gamma():
    if "GAMMA" in CONF_DICT.keys():
        return CONF_DICT["GAMMA"]
    else:
        return GAMMA
    
def get_random_seed():
    if "RANDOM_SEED" in CONF_DICT.keys():
        return CONF_DICT["RANDOM_SEED"]
    else:
        return RANDOM_SEED
    
def get_path():
    if "PATH" in CONF_DICT.keys():
        return CONF_DICT["PATH"]
    else:
        return PATH
    
def get_nlp_dataset_path():
    if "DATASET_PATH" in CONF_DICT.keys():
        return CONF_DICT["DATASET_PATH"]
    elif "DATA_FILENAME" in CONF_DICT.keys() and "PATH" in CONF_DICT.keys() :
        return CONF_DICT["PATH"]+CONF_DICT["DATA_FILENAME"]
    else:
        return PATH+DATA_FILENAME
    
def get_data_filename():
    if "DATA_FILENAME" in CONF_DICT.keys():
        return CONF_DICT["DATA_FILENAME"]
    else:
        return DATA_FILENAME
    
def dump_conf_dict(path):
    conf_dict = {
        "CLIP_MODEL_NAME": get_clip_model_name(),
        "OBJECT_TYPE_JSON_PATH": get_object_type_json_path(),
        "LOSS_CRITERION": get_loss_criterion(),
        "NUM_FOLDS": get_num_folds(),
        "PATH": get_path(),
        "DATA_FILENAME": get_data_filename(),
        "NUM_EPOCHS": get_num_epochs(),
        "NO_CROSSVAL": get_no_crossval(),
        "LEARNING_RATE": get_learning_rate(),
        "GAMMA": get_gamma(),
        "RANDOM_SEED": get_random_seed(),
        "MODEL_PATH": get_model_path(),
        "CROSSVAL_FOLD_LOADED": get_crossval_fold_loaded(),
        "CALCULATE_COSINE_SIMILARITY_MATRIX":  get_calculate_cosine_similarity_matrix(),
        "FROZEN_CLIP": get_frozen_clip()
    }

    conf_dict = custom_json_serializer(conf_dict)

    with open(path+'conf_dict.json', 'w') as f:
        json.dump(conf_dict, f)

def load_conf_dict(path):
    f = open(path)

    global CONF_DICT 
    CONF_DICT = custom_json_deserializer(json.load(f))