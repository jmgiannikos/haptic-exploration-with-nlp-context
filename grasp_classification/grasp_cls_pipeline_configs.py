import torch
import torch.nn as nn

PROVIDE_RAW_NLP_PROMPT = True
LOAD_COLOR = True
CALCULATE_LANGUAGE_EMBEDDING = True
BATCH_SIZE = 20
NUM_EPOCHS = 50
VERBOSE = True
PIXEL_REDUCTION_FACTOR = 2
CURRENT_DEVICE = "drax"
PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/",
        "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/"}
DATA_DIRECTORY = {"laptop": "ico_sphere_dataset/",
                  "drax": "irl_mixed_dataset/"} # "block_dataset/ cube_dataset/ cylinder1_dataset/ cylinder2_dataset/ ico_sphere_dataset/"
NO_CROSSVAL = False
MODEL_PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/block training model snapshots/_model_snapshot_70",
              "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset nlp classifiers 20ep/v3 batch norm l-tags color_model_snapshot_20_fold_0"}

# the appropriate val file paths should be saved in wandb run and easily retrievable. Have to be set before eval
VAL_FILE_PATHS = {"laptop":"",
                  "drax":["/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_ico_sphere_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_ico_sphere_semi_random2.npz"]}

"""
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere4.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere6.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere7.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere8.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere9.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere10.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere11.npz",
                          "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/real_grasp__ico-sphere12.npz",


"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_0_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_1_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_2_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_3_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_4_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_5_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_6_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_7_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_8_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_9_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_10_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_11_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_12_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_13_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_14_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_15_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_16_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_17_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_18_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_19_fold_0",
                       "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/full dataset baseline 20ep/v3 batch norm color 2_model_snapshot_20_fold_0",
"""

"""
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_block_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_block_semi_random2.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cube_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cube_semi_random2.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cylinder_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cylinder_semi_random2.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cylinder2_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_cylinder2_semi_random2.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_ico_sphere_semi_random.npz",
"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/testset/testset_ico_sphere_semi_random2.npz",
"""

TRAIN = False
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LANGUAGE_PROMPTS_PATH = {"drax":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/simple_nlp_prompts.npz",
                         "raw_drax":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/simple_nlp_use_dataset.npz"}
USE_LANGUAGE_PROMPTS = True
LEARNING_RATE = 0.002
CRITERION_ARGS = {"weight": torch.tensor([0.1,0.9]).to(DEVICE)}
LOSS_CRITERION = nn.NLLLoss(**CRITERION_ARGS)
CLIP_MODEL_NAME = "ViT-L/14"
GAMMA = 0.9
NLP_CLASSIFIER_PATH = {"vit-l":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/nlp_cls_results_run_07.17.23_23:30:35/nlp cls v2_best_model_fold_0",
                       "vit-b":"/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/nlp-cls/nlp_cls_results_run_07.17.23_22:42:12/nlp cls v1_best_model_fold_0"}
LOAD_CNN = False
CNN_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/results_run_07.19.23_20:44:38/v3 batch norm color_best_model_fold_0"
FROZEN_CNN = False
NLP_CLASSIFIER_TYPE = "vit-l"
DATASET_NAME = "mixed dataset"
FROZEN_CLIP = True
REGULAR_SAVE = False
EPOCH_SAVE = False
LOAD_PRETRAINED_MODEL = True


CONF_DICT = {}

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