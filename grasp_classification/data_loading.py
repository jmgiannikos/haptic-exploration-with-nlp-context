import numpy as np
import torch
import os

import grasp_cls_pipeline_configs as configs
import grasp_datasets as gds
import data_processing as dp
import grasp_cls_utils as gcls_utils
import nlp_datasets as nlpds

def nlp_train_test_split(dataset, labels, num_folds):
    dataset, labels = unison_shuffled_copies(dataset, labels)
    chunk_size = int(len(labels)/num_folds)
    crossval_dict = {}
    for fold in range(num_folds):
        start_idx = chunk_size*fold
        if fold != num_folds-1:
            end_idx = start_idx + chunk_size
        else:
            end_idx = -1

        crossval_dict[f"val_{fold}_data"] = dataset[start_idx:end_idx]
        crossval_dict[f"val_{fold}_labels"] = labels[start_idx:end_idx]
        if fold != num_folds-1:
            crossval_dict[f"train_{fold}_data"] = dataset[[idx for idx in range(len(labels)) if idx < start_idx or idx >= end_idx]]
            crossval_dict[f"train_{fold}_labels"] = labels[[idx for idx in range(len(labels)) if idx < start_idx or idx >= end_idx]]
        else:
            crossval_dict[f"train_{fold}_data"] = dataset[0:start_idx]
            crossval_dict[f"train_{fold}_labels"] = labels[0:start_idx]
    
    return crossval_dict

def load_nlp_dataset(dataset_path=""):
    dataset = np.load(dataset_path)

    data = dataset["nlp prompt"]
    data = dp.prune_dimensions(data)

    labels = dataset["object type"]

    return data, labels

def nlp_dataset_to_data_loader(dataset=None, batch_size=1, object_types=None):
    dataset = nlpds.nlp_dataset(dataset[0], dataset[1])
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def haptic_dataset_to_data_loader(dataset=None, data_path="", batch_size=1, object_types=None):
    if dataset is not None:
        if configs.get_load_color():
            dataset = gds.color_dataset(data=dataset[0], labels=dataset[1], object_types=object_types)
        else:
            dataset = gds.depth_dataset(data=dataset[0], labels=dataset[1], object_types=object_types)
    else:
        dataset = gds.depth_dataset(data_path=data_path, object_types=object_types)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def load_depth_dataset(dataset_path="", load_lift_dist=False):
    dataset = np.load(dataset_path)

    if not configs.get_load_color():
        data_key = "tactile_depth"
    else:
        data_key = "tactile_color"

    data = dataset[data_key]
    data = dp.prune_dimensions(data)

    if configs.get_load_color():
        data = np.transpose(data, axes=(0,1,4,2,3))
        data = data.astype('float32')

    if configs.get_pixel_reduction_factor() is not None:
        data = dp.reduce_depth_image_fidelity(data, reduction_factor_x=configs.get_pixel_reduction_factor(), reduction_factor_y=configs.get_pixel_reduction_factor())

    data = dp.normalize_min_max(data)

    labels = dataset["lift_success"]

    if not load_lift_dist:
        return data, labels
    else:
        pos_before_lift = dataset["after_grasp_pos"]
        pos_after_lift = dataset["final_pos"]
        lift_distances = pos_after_lift - pos_before_lift
        return data, labels, lift_distances

# loads all npz files in given folder
def get_dataset_paths(dataset_folder_path):
    files_list = os.listdir(dataset_folder_path)
    dataset_paths = [dataset_folder_path + filename for filename in files_list if gcls_utils.ends_in_npz(filename)]
    return dataset_paths

def split_file_list_evenly(file_list, fold_num, current_fold):
    distinct_object_type_dict = {}
    for name in file_list:
        identifier = name.split("/")[-1].split("_")[1]
        gcls_utils.dict_list_append(identifier, name, distinct_object_type_dict)

    val_list = []
    train_list = []
    for key in distinct_object_type_dict.keys():
        val_sublist, train_sublist = split_file_list(distinct_object_type_dict[key], fold_num, current_fold)
        val_list = val_list + val_sublist
        train_list = train_list + train_sublist

    return val_list, train_list

def split_file_list(file_list, fold_num, current_fold):
    block_size = int(len(file_list)/fold_num)
    start_idx = current_fold*block_size
    if current_fold == fold_num:
        end_idx = -1
    else:
        end_idx = start_idx+block_size

    val_list = file_list[start_idx:end_idx]
    train_list = [file for file in file_list if file not in val_list]

    return val_list, train_list

def unison_shuffled_copies(a, b, c=None):
    if c is not None:
        c = np.array(c)
        assert np.shape(a)[0] == np.shape(b)[0] and np.shape(a)[0] == np.shape(c)[0]
        p = np.random.permutation(np.shape(a)[0])
        return a[p], b[p], c[p]
    else:
        assert np.shape(a)[0] == np.shape(b)[0]
        p = np.random.permutation(np.shape(a)[0])
        return a[p], b[p]

def load_multi_file_dataset(file_names):
    object_types = []
    for i, train_file_name in enumerate(file_names):
        object_type = train_file_name.split("/")[-1].split("_")[1]

        if i == 0:
            train_data, train_labels = load_depth_dataset(dataset_path=train_file_name)

            train_data = dp.prune_dimensions(train_data)
            train_labels = dp.prune_dimensions(train_labels)

            add_len = np.shape(train_labels)[0]
        else:
            train_data_append, train_labels_append = load_depth_dataset(dataset_path=train_file_name)

            train_data_append = dp.prune_dimensions(train_data_append)
            train_labels_append = dp.prune_dimensions(train_labels_append)

            add_len = np.shape(train_labels_append)[0]

            train_data = np.append(train_data, train_data_append, 0)
            train_labels = np.append(train_labels, train_labels_append, 0)

        object_types = object_types + [object_type]*add_len

    train_data, train_labels, object_type = unison_shuffled_copies(train_data, train_labels, object_types)

    train_loader = haptic_dataset_to_data_loader(dataset=(train_data, train_labels), batch_size=configs.get_batch_size(), object_types = object_types)
    
    return train_loader

def split_file_list_evenly_groupings(file_list, fold_num, current_fold):
    distinct_object_type_dict = {}
    for name in file_list:
        identifier = name.split("/")[-1].split("_")[1]
        gcls_utils.dict_list_append(identifier, name, distinct_object_type_dict)

    val_list = []
    for i, key in enumerate(distinct_object_type_dict.keys()):
        val_sublist, train_sublist = split_file_list(distinct_object_type_dict[key], fold_num, current_fold)
        val_list = val_list + val_sublist
        if i == 0:
            train_list = np.array([train_sublist])
        else:
            train_list = np.append(train_list, np.array([train_sublist]), 0)

    train_list = train_list.T

    return val_list, train_list