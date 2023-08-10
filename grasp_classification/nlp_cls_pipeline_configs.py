import torch.nn as nn
import torch

CLIP_MODEL_NAME = "ViT-B/32"
OBJECT_TYPE_JSON_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/object_types.json"
LOSS_CRITERION = nn.CrossEntropyLoss()
NUM_FOLDS = 3
PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/"
DATA_FILENAME = "simple_nlp_dataset.npz"
NUM_EPOCHS = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NO_CROSSVAL = False
LEARNING_RATE = 0.0002
GAMMA = 0.99
RANDOM_SEED = 42

CONF_DICT = {}

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
    else:
        return PATH+DATA_FILENAME