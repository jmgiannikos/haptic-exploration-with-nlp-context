import torch
import torch.nn as nn

PROVIDE_RAW_NLP_PROMPT = True
LOAD_COLOR = True
CALCULATE_LANGUAGE_EMBEDDING = True
BATCH_SIZE = 20
NUM_EPOCHS = 50
VERBOSE = False
PIXEL_REDUCTION_FACTOR = 2
CURRENT_DEVICE = "laptop"
PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/",
        "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/"}
DATA_DIRECTORY = {"laptop": "grasps/",
                  "drax": "irl_mixed_dataset/"} # "block_dataset/ cube_dataset/ cylinder1_dataset/ cylinder2_dataset/ ico_sphere_dataset/"
NO_CROSSVAL = False
MODEL_PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/results_run_09.12.23_10:33:57/v3 batch norm color 2_best_model_fold_4",
              "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset nlp classifiers 20ep/v3 batch norm l-tags color_model_snapshot_20_fold_0"}

# the appropriate val file paths should be saved in wandb run and easily retrievable. Have to be set before eval
VAL_FILE_PATHS = {"laptop":["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_rubber_duck_semi_random12.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_rubber_duck_semi_random2.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_microphone1_semi_random7.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_microphone1_semi_random9.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_hair_dryer_semi_random11.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_hair_dryer_semi_random8.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_can1_semi_random12.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_can1_semi_random6.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_bottle1_semi_random12.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_bottle1_semi_random4.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_camera_semi_random6.npz",
"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/grasp_camera_semi_random7.npz"],
                  "drax":[]}

TRAIN = True
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LANGUAGE_PROMPTS_PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/gpt_generated_object_prompt_dataset.npz",
                        "raw_laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/gpt_generated_holdout_prompts.npz",
                         "drax":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/simple_nlp_prompts.npz",
                         "raw_drax":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_use_dataset.npz"}
USE_LANGUAGE_PROMPTS = True
LEARNING_RATE = 0.002
CRITERION_ARGS = {"weight": torch.tensor([0.05,0.95]).to(DEVICE)}
LOSS_CRITERION = nn.NLLLoss(**CRITERION_ARGS)
CLIP_MODEL_NAME = "ViT-L/14"
GAMMA = 0.9
NLP_CLASSIFIER_PATH = {"vit-l":"",
                       "vit-b":"/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/nlp_cls_results_run_09.12.23_03:41:28/nlp cls complex obj_best_model_fold_0"}
LOAD_CNN = False
CNN_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/results_run_07.19.23_20:44:38/v3 batch norm color_best_model_fold_0"
FROZEN_CNN = False
NLP_CLASSIFIER_TYPE = "vit-b"
DATASET_NAME = "complex object dataset"
FROZEN_CLIP = True
REGULAR_SAVE = True
EPOCH_SAVE = False
LOAD_PRETRAINED_MODEL = False
PREDEFINED_TRAIN_VAL_PATH = "/home/jan-malte/Bachelors Thesis/predefined_train_test_splits.json"
PREDEFINED_LANGUAGE_SPLITS = "/home/jan-malte/Bachelors Thesis/predefined_nlp_dataset.json"
USE_RAW_CLIP_MODEL = True

CONF_DICT = {}

def get_use_raw_clip_model():
    if "USE_RAW_CLIP_MODEL" in CONF_DICT.keys():
        return CONF_DICT["USE_RAW_CLIP_MODEL"]
    else:
        return USE_RAW_CLIP_MODEL

def get_predefined_language_splits_path():
    if "PREDEFINED_LANGUAGE_SPLITS" in CONF_DICT.keys():
        return CONF_DICT["PREDEFINED_LANGUAGE_SPLITS"]
    else:
        return PREDEFINED_LANGUAGE_SPLITS

def get_predefined_train_val_path():
    if "PREDEFINED_TRAIN_VAL_PATH" in CONF_DICT.keys():
        return CONF_DICT["PREDEFINED_TRAIN_VAL_PATH"]
    else:
        return PREDEFINED_TRAIN_VAL_PATH
    
def get_load_pretrained_model():
    if "LOAD_PRETRAINED_MODEL" in CONF_DICT.keys():
        return CONF_DICT["LOAD_PRETRAINED_MODEL"]
    else:
        return LOAD_PRETRAINED_MODEL

def get_epoch_save():
    if "EPOCH_SAVE" in CONF_DICT.keys():
        return CONF_DICT["EPOCH_SAVE"]
    else:
        return EPOCH_SAVE

def get_regular_save():
    if "REGULAR_SAVE" in CONF_DICT.keys():
        return CONF_DICT["REGULAR_SAVE"]
    else:
        return REGULAR_SAVE

def get_frozen_clip():
    if "FROZEN_CLIP" in CONF_DICT.keys():
        return CONF_DICT["FROZEN_CLIP"]
    else:
        return FROZEN_CLIP

def get_dataset_name():
    if "DATASET_NAME" in CONF_DICT.keys():
        return CONF_DICT["DATASET_NAME"]
    else:
        return DATASET_NAME

def get_frozen_cnn():
    if "FROZEN_CNN" in CONF_DICT.keys():
        return CONF_DICT["FROZEN_CNN"]
    else:
        return FROZEN_CNN

def get_cnn_path():
    if "CNN_PATH" in CONF_DICT.keys():
        return CONF_DICT["CNN_PATH"]
    else:
        return CNN_PATH

def get_load_cnn():
    if "LOAD_CNN" in CONF_DICT.keys():
        return CONF_DICT["LOAD_CNN"]
    else:
        return LOAD_CNN
    
def get_cnn():
    cnn_path = get_cnn_path()

    cnn = torch.load(cnn_path).cnn_feature_extract
    return cnn

def get_provide_raw_nlp_prompt():
    if "PROVIDE_RAW_NLP_PROMPT" in CONF_DICT.keys():
        return CONF_DICT["PROVIDE_RAW_NLP_PROMPT"]
    else:
        return PROVIDE_RAW_NLP_PROMPT

def get_nlp_classifier_path():
    if "NLP_CLASSIFIER_PATH" in CONF_DICT.keys():
        return CONF_DICT["NLP_CLASSIFIER_PATH"]
    else:
        return NLP_CLASSIFIER_PATH[NLP_CLASSIFIER_TYPE]

def get_use_language_prompts():
    if "USE_LANGUAGE_PROMPTS" in CONF_DICT.keys():
        return CONF_DICT["USE_LANGUAGE_PROMPTS"]
    else:
        return USE_LANGUAGE_PROMPTS

def get_gamma():
    if "GAMMA" in CONF_DICT.keys():
        return CONF_DICT["GAMMA"]
    else:
        return GAMMA

def get_criterion_args():
    if "CRITERION_ARGS" in CONF_DICT.keys():
        return CONF_DICT["CRITERION_ARGS"]
    else:
        return CRITERION_ARGS

def get_load_color():
    if "LOAD_COLOR" in CONF_DICT.keys():
        return CONF_DICT["LOAD_COLOR"]
    else:
        return LOAD_COLOR
    
def get_calculate_language_embedding():
    if "CALCULATE_LANGUAGE_EMBEDDING" in CONF_DICT.keys():
        return CONF_DICT["CALCULATE_LANGUAGE_EMBEDDING"]
    else:
        return CALCULATE_LANGUAGE_EMBEDDING
    
def get_batch_size():
    if "BATCH_SIZE" in CONF_DICT.keys():
        return CONF_DICT["BATCH_SIZE"]
    else:
        return BATCH_SIZE
    
def get_num_epochs():
    if "NUM_EPOCHS" in CONF_DICT.keys():
        return CONF_DICT["NUM_EPOCHS"]
    else:
        return NUM_EPOCHS
    
def get_verbose():
    if "VERBOSE" in CONF_DICT.keys():
        return CONF_DICT["VERBOSE"]
    else:
        return VERBOSE
    
def get_pixel_reduction_factor():
    if "PIXEL_REDUCTION_FACTOR" in CONF_DICT.keys():
        return CONF_DICT["PIXEL_REDUCTION_FACTOR"]
    else:
        return PIXEL_REDUCTION_FACTOR
    
def get_current_device():
    if "CURRENT_DEVICE" in CONF_DICT.keys():
        return CONF_DICT["CURRENT_DEVICE"]
    else:
        return CURRENT_DEVICE
    
def get_path():
    if "PATH" in CONF_DICT.keys():
        return CONF_DICT["PATH"]
    else:
        return PATH[get_current_device()]
    
def get_no_crossval():
    if "NO_CROSSVAL" in CONF_DICT.keys():
        return CONF_DICT["NO_CROSSVAL"]
    else:
        return NO_CROSSVAL
    
def get_model_path():
    if "MODEL_PATH" in CONF_DICT.keys():
        return CONF_DICT["MODEL_PATH"]
    else:
        return MODEL_PATH[get_current_device()]
    
def get_val_file_paths():
    if "VAL_FILE_PATHS" in CONF_DICT.keys():
        return CONF_DICT["VAL_FILE_PATHS"]
    else:
        return VAL_FILE_PATHS[get_current_device()]
    
def get_train():
    if "TRAIN" in CONF_DICT.keys():
        return CONF_DICT["TRAIN"]
    else:
        return TRAIN
    
def get_num_folds():
    if "NUM_FOLDS" in CONF_DICT.keys():
        return CONF_DICT["NUM_FOLDS"]
    else:
        return NUM_FOLDS
    
def get_device():
    if "DEVICE" in CONF_DICT.keys():
        return CONF_DICT["DEVICE"]
    else:
        return DEVICE
    
def get_language_prompts_path():   
    if "LANGUAGE_PROMPTS_PATH" in CONF_DICT.keys():
        return CONF_DICT["LANGUAGE_PROMPTS_PATH"]
    else:
        if get_provide_raw_nlp_prompt():
            return LANGUAGE_PROMPTS_PATH[f"raw_{get_current_device()}"]
        else:
            return LANGUAGE_PROMPTS_PATH[get_current_device()]
    
def get_dataset_path():
    if "DATASET_PATH" in CONF_DICT.keys():
        return CONF_DICT["DATASET_PATH"]
    else:
        return PATH[get_current_device()]+DATA_DIRECTORY[get_current_device()]

def get_learning_rate():
    if "LEARNING_RATE" in CONF_DICT.keys():
        return CONF_DICT["LEARNING_RATE"]
    else:
        return LEARNING_RATE  

def get_loss_criterion():
    if "LOSS_CRITERION" in CONF_DICT.keys():
        return CONF_DICT["LOSS_CRITERION"]
    else:
        return LOSS_CRITERION
    
def get_clip_model_name():
    if "CLIP_MODEL_NAME" in CONF_DICT.keys():
        return CONF_DICT["CLIP_MODEL_NAME"]
    else:
        return CLIP_MODEL_NAME