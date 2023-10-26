import data_loading as dl
import grasp_cls_pipeline_configs as configs
import json
import nlp_processing as nlpp
import numpy as np
import grasp_cls_utils as gcls_utils
import generate_nlp_dataset as nlp_gen

PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/grasps/"
NLP_JSON_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/grasp_classification/gpt_generated_object_prompt_dataset.json"
PRETRAIN_DATA_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/gpt_generated_pretrain_prompts.npz"
HOLDOUT_DATASET_PATH =  "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/gpt_generated_holdout_prompts.npz"
VAL_FILE_NAMES_DICT = {
    0:["grasp_camera_semi_random7.npz",
       "grasp_camera_semi_random9.npz",
       "grasp_bottle1_semi_random2.npz",
       "grasp_bottle1_semi_random9.npz",
       "grasp_rubber_duck_semi_random11.npz",
       "grasp_rubber_duck_semi_random7.npz",
       "grasp_hair_dryer_semi_random7.npz",
       "grasp_hair_dryer_semi_random9.npz",
       "grasp_can1_semi_random10.npz",
       "grasp_can1_semi_random5.npz",
       "grasp_microphone1_semi_random.npz",
       "grasp_microphone1_semi_random3.npz"],
    1:["grasp_camera_semi_random12.npz",
       "grasp_camera_semi_random4.npz",
       "grasp_bottle1_semi_random8.npz",
       "grasp_bottle1_semi_random11.npz",
       "grasp_rubber_duck_semi_random4.npz",
       "grasp_rubber_duck_semi_random6.npz",
       "grasp_hair_dryer_semi_random5.npz",
       "grasp_hair_dryer_semi_random12.npz",
       "grasp_can1_semi_random3.npz",
       "grasp_can1_semi_random6.npz",
       "grasp_microphone1_semi_random8.npz",
       "grasp_microphone1_semi_random12.npz"],
    2:["grasp_camera_semi_random10.npz",
       "grasp_camera_semi_random11.npz",
       "grasp_bottle1_semi_random7.npz",
       "grasp_bottle1_semi_random6.npz",
       "grasp_rubber_duck_semi_random12.npz",
       "grasp_rubber_duck_semi_random9.npz",
       "grasp_hair_dryer_semi_random.npz",
       "grasp_hair_dryer_semi_random2.npz",
       "grasp_can1_semi_random12.npz",
       "grasp_can1_semi_random2.npz",
       "grasp_microphone1_semi_random11.npz",
       "grasp_microphone1_semi_random7.npz"],
    3:["grasp_camera_semi_random5.npz",
       "grasp_camera_semi_random.npz",
       "grasp_bottle1_semi_random4.npz",
       "grasp_bottle1_semi_random3.npz",
       "grasp_rubber_duck_semi_random8.npz",
       "grasp_rubber_duck_semi_random10.npz",
       "grasp_hair_dryer_semi_random6.npz",
       "grasp_hair_dryer_semi_random3.npz",
       "grasp_can1_semi_random4.npz",
       "grasp_can1_semi_random7.npz",
       "grasp_microphone1_semi_random9.npz",
       "grasp_microphone1_semi_random2.npz"],
    4:["grasp_camera_semi_random8.npz",
       "grasp_camera_semi_random6.npz",
       "grasp_bottle1_semi_random10.npz",
       "grasp_bottle1_semi_random12.npz",
       "grasp_rubber_duck_semi_random.npz",
       "grasp_rubber_duck_semi_random5.npz",
       "grasp_hair_dryer_semi_random11.npz",
       "grasp_hair_dryer_semi_random8.npz",
       "grasp_can1_semi_random.npz",
       "grasp_can1_semi_random11.npz",
       "grasp_microphone1_semi_random4.npz",
       "grasp_microphone1_semi_random10.npz"]
}

def remove_val_from_dataset_paths(dataset_paths, val):
    train_datasets = []
    for name in dataset_paths:
        if name not in val:
            train_datasets.append(name)
    return train_datasets

def assemble_full_path(name_list, path):
    paths = []
    for name in name_list:
        paths.append(path+name)
    return paths

def make_predefined_train_test_splits():
    dataset_paths = dl.get_dataset_paths(configs.get_dataset_path())
    splits_by_fold = {}
    for fold in range(5):
        val_files = assemble_full_path(VAL_FILE_NAMES_DICT[fold], PATH)
        train_files_dataset = remove_val_from_dataset_paths(dataset_paths, val_files)
        
        distinct_object_type_dict = {}
        for name in train_files_dataset:
            identifier = name.split("/")[-1].split("_")[1]
            gcls_utils.dict_list_append(identifier, name, distinct_object_type_dict)

        for i, key in enumerate(distinct_object_type_dict.keys()):
            train_sublist = distinct_object_type_dict[key]
            if i == 0:
                train_list = np.array([train_sublist])
            else:
                train_list = np.append(train_list, np.array([train_sublist]), 0)

        train_list = train_list.T.tolist()

        splits_by_fold[fold] = {"val":val_files,
                                "train":train_list}
    file = open("predefined_train_test_splits.json", "w")
    json.dump(splits_by_fold, file)

def make_predefined_nlp_dataset(language_prompts_path):
    train_val_prompts = nlpp.split_nlp_prompt_dict(configs.get_num_folds(), language_prompts_path=language_prompts_path)

    for fold in range(configs.get_num_folds()):
        for key in train_val_prompts[fold]["val"].keys():
            if not isinstance(train_val_prompts[fold]["val"][key], list):
                train_val_prompts[fold]["val"][key] = train_val_prompts[fold]["val"][key].tolist()
    
    file = open("predefined_nlp_dataset.json", "w")
    json.dump(train_val_prompts, file)

def test_predefined_train_test_splits():
    file = open("predefined_train_test_splits.json", "r")
    data = json.load(file)
    for fold in range(configs.get_num_folds()):
        val_files = data[f"{fold}"]["val"]
        train_files = data[f"{fold}"]["train"]
        for val_file_name in val_files:
            for block in train_files:
                if val_file_name in block:
                    print(f"error: val file {val_file_name} is in training set")

def make_pretraining_holdout_dataset(ratio=0.7):
    file = open(NLP_JSON_PATH, "r")
    data = json.load(file)

    holdout_data = {}
    pretrain_data = {}
    for object_type in data.keys():
        holdout_end_idx = int(len(data[object_type])*ratio)
        holdout_list = data[object_type][0:holdout_end_idx]
        pretrain_list = data[object_type][holdout_end_idx:-1]
        holdout_data[object_type] = holdout_list
        pretrain_data[object_type] = pretrain_list

    nlp_gen.make_train_dataset(pretrain_data, PRETRAIN_DATA_PATH)
    nlp_gen.make_use_dataset(HOLDOUT_DATASET_PATH, holdout_data)

def main():
    make_pretraining_holdout_dataset()
    make_predefined_nlp_dataset(HOLDOUT_DATASET_PATH)
    make_predefined_train_test_splits()
    test_predefined_train_test_splits()

if __name__ == '__main__':
    main()