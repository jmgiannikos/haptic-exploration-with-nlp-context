import torch
#import torchvision
#import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
import torch.optim as optim
import statistics
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import wandb
from datetime import datetime
import clip

CALCULATE_LANGUAGE_EMBEDDING = False
BATCH_SIZE = 20
NUM_EPOCHS = 8
VERBOSE = False
PIXEL_REDUCTION_FACTOR = 2
CURRENT_DEVICE = "drax"
PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/",
        "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/"}
DATA_DIRECTORY = {"laptop": "grasp_datasets/",
        "drax": "mixed_object_dataset/"}
NO_CROSSVAL = False
MODEL_PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/block training model snapshots/_model_snapshot_70",
              "drax": "/home/jan-malte/haptic-exploration-with-nlp-context/simulation/_model_snapshot_70"}

# the appropriate val file paths should be saved in wandb run and easily retrievable. Have to be set before eval
VAL_FILE_PATHS = {"laptop":"",
            "drax":""}
TRAIN = True
NUM_FOLDS = 5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LANGUAGE_PROMPTS_PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/simple_nlp_prompts.npz"

class Depth_Grasp_Classifier_v3_nrm_ltagp(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(5189, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm l-tags-precalc"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3_norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.BatchNorm2d(576),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(144),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5184, 1296),
            nn.BatchNorm1d(1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 batch norm"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3l(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5696, 1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3 language"

    def forward(self, x, nlp_embedding):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.cat((x,nlp_embedding),1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v3(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 1),
            nn.Conv2d(32, 576, kernel_size=9, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(576, 144, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(5184, 1296),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1296, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v3"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v2_w(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1728, 864),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(864, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class Depth_Grasp_Classifier_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1728, 864),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(864, 2),
            torch.nn.LogSoftmax(1)
        )

        self.name = "v2"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# neural network that classifiers wether or not the grasp was successful based on the depth image
class Depth_Grasp_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_feature_extract = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 128, kernel_size=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 4608),
            nn.ReLU(),
            nn.Linear(4608, 2),
            torch.nn.LogSoftmax(1)
        )
        self.name = "v1"

    def forward(self, x):
        x = self.cnn_feature_extract(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classifier(x)
        return x
    
class depth_dataset(Dataset):
    def __init__(self, data=None, labels=None, data_path=None, transform=None, target_transform=None, object_types=None):

        global_index = 0
        global_index_dict = {}
        if data is None:
            # EXTRACT LABELS AND RECORD INDEX BOUNDARIES
            while True:
                if i == 1:
                    file_suffix = ""    
                    path = data_path + file_suffix + ".npz"

                    if os.path.isfile(path):
                        dataset = np.load(path)
                        labels = dataset["lift_success"] 
                        global_index = add_to_global_index_dict(global_index_dict, global_index, len(labels), path)
                    else:
                        break

                else:
                    file_suffix = str(i)
                
                    path = data_path + file_suffix + ".npz"

                    if os.path.isfile(path):
                        dataset = np.load(path) 
                        if np.shape(dataset["lift_success"])[0] == 1:
                            labels = np.appen(labels, dataset["lift_success"][0], 0)
                        else:
                            labels = np.appen(labels, dataset["lift_success"], 0)

                        global_index = add_to_global_index_dict(global_index_dict, global_index, len(labels), path)
                    else:
                        break

                i += 1

            self.depth_labels = convert_to_class_idx(labels)
            self.data_path = data_path
            self.global_index_dict = global_index_dict
            self.depth_images = None
        else:
            self.depth_labels = convert_to_class_idx(labels)
            self.depth_images = data

        self.object_types = object_types

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.depth_labels)

    def __getitem__(self, idx):
        if self.depth_images is None:
            path, local_index = self.global_index_dict[idx]
            dataset = np.load(path)
            depth_images = dataset["tactile_depth"]
            if np.shape(depth_images)[0] == 1:
                depth_image = depth_images[0][local_index]
            else:
                depth_image = depth_images[local_index]

        else:
            depth_image = self.depth_images[idx]
            depth_label = self.depth_labels[idx]
            if self.transform:
                depth_image = self.transform(depth_image)
            if self.target_transform:
                depth_label = self.target_transform(depth_label)
            if self.object_types is not None:
                object_type = self.object_types[idx]
                return depth_image, depth_label, object_type
            else:
                return depth_image, depth_label

def render_depth(depth, axsimg=None):
    if axsimg is None:
        depth = prune_dimensions(depth)
        fig, axs = plt.subplots(2)
        axsimg = [None, None, None, None]
        axsimg[0] = axs[0].imshow(depth[0], cmap='gist_gray')
        axsimg[1] = axs[1].imshow(depth[1], cmap='gist_gray')
    else:
        axsimg[0].set_data(depth[0])
        axsimg[1].set_data(depth[1])

    plt.draw()
    plt.show(block=True)
    return axsimg

def add_to_global_index_dict(index_dict, start_index, end_offset, file_path):
    for i in range(end_offset):
        index_dict[start_index+i] = (file_path, i)
    return end_offset

def convert_to_class_idx(labels):
    class_indices = []
    for label in labels:
        if label:
            class_indices.append(1)
        else:
            class_indices.append(0)
    return np.asarray(class_indices)

def dataset_to_data_loader(dataset=None, data_path="", batch_size=1, object_types=None):
    if dataset is not None:
        dataset = depth_dataset(data=dataset[0], labels=dataset[1], object_types=object_types)
    else:
        dataset = depth_dataset(data_path=data_path, object_types=object_types)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def load_depth_dataset(dataset_path="", normalize=True):
    dataset = np.load(dataset_path)
    if PIXEL_REDUCTION_FACTOR is None:
        data = dataset["tactile_depth"]
    else:
        tactile_depth = prune_dimensions(dataset["tactile_depth"])
        data = reduce_depth_image_fidelity(tactile_depth, reduction_factor_x=PIXEL_REDUCTION_FACTOR, reduction_factor_y=PIXEL_REDUCTION_FACTOR)
    
    labels = dataset["lift_success"]

    if normalize:
        initialized = False
        for data_point in data:
            dp_min = np.amin(data_point)
            dp_max = np.amax(data_point)
            if (dp_max-dp_min) != 0:
                data_point = (data_point-dp_min)/(dp_max-dp_min)

            if not initialized:
                new_data_points = np.array([data_point])
                initialized = True
            else:
                new_data_points = np.append(new_data_points, [data_point], 0)

        data = new_data_points

    return data, labels

def get_crossval_idxs(data, labels):
    splitter = StratifiedKFold()
    splits = splitter.split(X=data, y=labels)
    return splits

def reduce_depth_image_fidelity(depth_images, reduction_factor_x=2, reduction_factor_y=2, verbose=VERBOSE):
    pool_function = nn.AvgPool2d((reduction_factor_x, reduction_factor_y), stride=(reduction_factor_x, reduction_factor_y))
    pooled_depth_images = None
    i = 0
    for depth_image in depth_images:
        finger_1 = torch.from_numpy(np.array([depth_image[0]]))
        finger_2 =  torch.from_numpy(np.array([depth_image[1]]))
        pooled_finger_1 = np.asarray(pool_function(finger_1))
        pooled_finger_2 =  np.asarray(pool_function(finger_2))
        pooled_fingers = np.asarray([np.append(pooled_finger_1, pooled_finger_2, 0)])

        if verbose and i < 10:
            render_depth(pooled_fingers) 
            i += 1

        if pooled_depth_images is None:
            pooled_depth_images = pooled_fingers
        else:
            pooled_depth_images = np.append(pooled_depth_images, pooled_fingers, 0)

    return pooled_depth_images

def validate_classifier(model, val_files, result_dict, show_misclassified=True, fold=-1, save_figure=True, results_path="", nlp_prompts=None, clip_model = None, object_type_dict=None):
    model.eval()
    tmp_val_dict = {}
    for val_iter, val_file_name in enumerate(val_files):
        val_data, val_labels = load_depth_dataset(dataset_path=val_file_name)
        val_data = prune_dimensions(val_data)
        val_labels = prune_dimensions(val_labels)
        val_loader = dataset_to_data_loader(dataset=(val_data, val_labels))

        pred_labels = []
        gt_labels = []
        pred_pos_probabilities = []
        j = 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        object_type = val_file_name.split("/")[-1].split("_")[1]

        for idx, data in enumerate(val_loader, 0):
            inputs, label = data
            inputs = inputs.to(DEVICE)
            gt_labels.append(label.detach().numpy()[0])

            if nlp_prompts is not None:
                if not CALCULATE_LANGUAGE_EMBEDDING:
                    nlp_embedding = object_type_dict[object_type][None,:]
                else:
                    nlp_embedding = get_language_embedding(nlp_prompts=nlp_prompts, object_type=object_type, clip_model=clip_model)
                prediction = model(inputs, nlp_embedding)
            else:
                prediction = model(inputs)

            prediction = prune_dimensions(prediction.cpu().detach().numpy())
            prediction = np.exp(prediction)

            pred_pos_probability = prediction[1]

            pred_pos_probabilities.append(pred_pos_probability)

            max_pred = -float("inf")
            max_pred_idx = None
            i = 0
            for pred in prediction:
                if pred > max_pred:
                    max_pred = pred
                    max_pred_idx = i
                i += 1

            if VERBOSE:
                if j <= 10:
                    print("predicted positive probability:" + str(pred_pos_probability))
                    print("pred label:" + str(max_pred_idx))
                    print("gt label:" + str(label.item()))
                    render_depth(inputs)
                    j += 1

                if show_misclassified and max_pred_idx != label.item():
                    print("### misclassification ###")
                    print("predicted positive probability:" + str(pred_pos_probability))
                    print("pred label:" + str(max_pred_idx))
                    print("gt label:" + str(label.item()))
                    render_depth(inputs)

            pred_labels.append(max_pred_idx)

            if label == 1:
                if pred_labels[idx] == 1:
                    true_positives += 1
                elif pred_labels[idx] == 0:
                    false_negatives += 1
            elif label == 0:
                if pred_labels[idx] == 1:
                    false_positives += 1
                if pred_labels[idx] == 0:
                    true_negatives += 1

        if save_figure:
            display = PrecisionRecallDisplay.from_predictions(y_true=gt_labels, y_pred=pred_pos_probabilities)
            plt.savefig(results_path+"precision_recall_curve_fold_"+str(fold)+"_num_"+str(val_iter))
            if VERBOSE:
                plt.show(block=True)

        if true_positives+false_positives > 0:
            precision = true_positives/(true_positives+false_positives)
        else:
            precision = 0

        if true_positives+false_negatives > 0:
            recall = true_positives/(true_positives+false_negatives)
        else:
            recall = 0

        dict_list_append("precision", precision, tmp_val_dict)
        dict_list_append("recall", recall, tmp_val_dict)
        dict_list_append("tp", true_positives, tmp_val_dict)
        dict_list_append("tn", true_negatives, tmp_val_dict)
        dict_list_append("fp", false_positives, tmp_val_dict)
        dict_list_append("fn", false_negatives, tmp_val_dict)

    stack_validation_averages(result_dict, tmp_val_dict) # collapse temporary entries and add them to global results
    model.train()


def dict_list_append(key, val, target_dict):
    if key in target_dict.keys():
        target_dict[key].append(val)
    else:
        target_dict[key] = [val]

def ends_in_npz(filename):
    split_name = filename.split(".")
    return "npz" == split_name[-1]

# loads all npz files in given folder
def get_dataset_paths(dataset_folder_path):
    files_list = os.listdir(dataset_folder_path)
    dataset_paths = [dataset_folder_path + filename for filename in files_list if ends_in_npz(filename)]
    return dataset_paths

def prune_dimensions(array):
    if np.shape(array)[0] == 1:
        return array[0]
    else:
        return array

def split_file_list_evenly(file_list, fold_num, current_fold):
    distinct_object_type_dict = {}
    for name in file_list:
        identifier = name.split("/")[-1].split("_")[1]
        dict_list_append(identifier, name, distinct_object_type_dict)

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

def stack_validation_averages(val_dict, tmp_val_dict):
    for key in tmp_val_dict.keys():
        if key == "precision" or key == "recall":
            value = sum(tmp_val_dict[key])/len(tmp_val_dict[key])
        else:
            value = sum(tmp_val_dict[key])
    
        dict_list_append(key, value, val_dict)

def info_dict_to_wandb_format(dictionary):
    for key in dictionary.keys():
        if isinstance(dictionary[key], list):
            if len(dictionary[key]) == 1:
                dictionary[key] = dictionary[key][0]
    return dictionary 

def get_language_embedding(nlp_prompts, object_type, clip_model):
    nlp_promt = random.choice(nlp_prompts[object_type])
    prompt_tokens = clip.tokenize(nlp_promt).to(DEVICE)
    prompt_embedding = clip_model.encode_text(prompt_tokens)
    return prompt_embedding

def unison_shuffled_copies(a, b, c):
    c = np.array(c)
    assert np.shape(a)[0] == np.shape(b)[0] and np.shape(a)[0] == np.shape(c)[0]
    p = np.random.permutation(np.shape(a)[0])
    return a[p], b[p], c[p]

def load_multi_file_dataset(file_names):
    object_types = []
    for i, train_file_name in enumerate(file_names):
        object_type = train_file_name.split("/")[-1].split("_")[1]

        if i == 0:
            train_data, train_labels = load_depth_dataset(dataset_path=train_file_name)

            train_data = prune_dimensions(train_data)
            train_labels = prune_dimensions(train_labels)

            add_len = np.shape(train_labels)[0]
        else:
            train_data_append, train_labels_append = load_depth_dataset(dataset_path=train_file_name)

            train_data_append = prune_dimensions(train_data_append)
            train_labels_append = prune_dimensions(train_labels_append)

            add_len = np.shape(train_labels_append)[0]

            train_data = np.append(train_data, train_data_append, 0)
            train_labels = np.append(train_labels, train_labels_append, 0)

        object_types = object_types + [object_type]*add_len

    train_data, train_labels, object_type = unison_shuffled_copies(train_data, train_labels, object_types)

    train_loader = dataset_to_data_loader(dataset=(train_data, train_labels), batch_size=BATCH_SIZE, object_types = object_types)
    
    return train_loader

def split_file_list_evenly_groupings(file_list, fold_num, current_fold):
    distinct_object_type_dict = {}
    for name in file_list:
        identifier = name.split("/")[-1].split("_")[1]
        dict_list_append(identifier, name, distinct_object_type_dict)

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

def generate_object_type_dict(dataset_paths):
    type_dict = {}
    index = 0
    for name in dataset_paths:
        object_type = name.split("/")[-1].split("_")[1]
        if object_type not in type_dict.keys():
            type_dict[object_type] = index
            index += 1

    for key in type_dict:
        idx_list = [0]*len(list(type_dict.keys()))
        idx_list[type_dict[key]] = 1
        idx_tensor = torch.tensor(idx_list).to(DEVICE)
        type_dict[key] = idx_tensor

    return type_dict


def train_test_depth_pipeline(dataset_path="", dnt_start="none", results_path="", regular_save=False, language_prompts_path=None):
    learning_rate = 0.001
    nll_weights = torch.tensor([0.142,1.0]).to(DEVICE)
    criterion = nn.NLLLoss(weight=nll_weights)
    clip_nlp_model = None
    if language_prompts_path is not None:
        clip_model_name = "ViT-B/32"
        clip_nlp_model, _ = clip.load(clip_model_name, device=DEVICE)
        nlp_promts = np.load(language_prompts_path)

    dataset_paths = get_dataset_paths(dataset_path)
    object_type_dict = generate_object_type_dict(dataset_paths)

    random.shuffle(dataset_paths)

    validation_results = {}
    plot_count = 1
    group_name = results_path.split("/")[-2]
    for fold in range(NUM_FOLDS):
        loss_vals_gtf = []
        loss_vals_gtt = []
        average_loss = []
        snapshot_count = 0
        current_best_f1 = -1 # always save at least first model
        gamma=0.8
        depth_grasp_classifier = Depth_Grasp_Classifier_v3_norm()
        depth_grasp_classifier.to(DEVICE)
        optimizer = optim.SGD(depth_grasp_classifier.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        val_files, train_files = split_file_list_evenly_groupings(dataset_paths, NUM_FOLDS, fold)

        config={
            "optimizer type": type(optimizer),
            "loss type": type(criterion),
            "learning_rate": learning_rate,
            "model": depth_grasp_classifier.name,
            "training datasets": train_files,
            "validation datasets": val_files,
            "epochs": NUM_EPOCHS,
            "crossval fold": fold,
            "total folds": NUM_FOLDS,
            "pixel reduction factor": PIXEL_REDUCTION_FACTOR,
            "start time and date": dnt_start,
            "scheduler type": type(scheduler)
        }

        if language_prompts_path is not None:
            config["clip model"] = clip_model_name
            config["language prompts"] = language_prompts_path

        if isinstance(scheduler, optim.lr_scheduler.ExponentialLR):
            config["scheduler gamma"] = gamma

        if isinstance(criterion, nn.NLLLoss):
            config["NLL weights"] = nll_weights

        run_name = "fold_" + str(fold)

        wandb.init(
            # set the wandb project where this run will be logged
            project="haptic-exploration-with-nlp",
            # track hyperparameters and run metadata
            config=config,
            name=run_name,
            group=group_name
        )

        print(f"###### fold: {fold} ######")

        depth_grasp_classifier.train()

        for epoch in range(NUM_EPOCHS):
            for file_idx, multi_file_train_chunk in enumerate(train_files):
                print(f"--- {multi_file_train_chunk} ---")

                train_loader = load_multi_file_dataset(multi_file_train_chunk)

                running_loss = 0
                
                for i, batch in enumerate(train_loader, 0):
                    inputs, labels, object_types = batch
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    if clip_nlp_model is not None:
                        if not CALCULATE_LANGUAGE_EMBEDDING:
                            for obj_type_idx, object_type in enumerate(object_types):
                                if obj_type_idx == 0:
                                    language_embedding = object_type_dict[object_type][None,:]
                                else:
                                    language_embedding = torch.cat((language_embedding, object_type_dict[object_type][None,:]), 0)
                        else:
                            language_embedding = get_language_embedding(nlp_prompts=nlp_promts, object_type=object_types, clip_model=clip_nlp_model)
                        
                        outputs = depth_grasp_classifier(inputs, language_embedding)
                    else:
                        outputs = depth_grasp_classifier(inputs)
                        
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    current_loss = loss.item()
                    running_loss += current_loss
                    #if i % 10 == 9:    # print every 2000 mini-batches
                    if VERBOSE:
                        print(f'[{epoch + 1}, {i + 1:5d}] loss: {current_loss:.3f}')
                    if BATCH_SIZE == 1:
                        if labels[0].item() == 0:
                            loss_vals_gtf.append(current_loss)
                            wandb.log({"ground truth negative loss": current_loss})
                        else:
                            loss_vals_gtt.append(current_loss)
                            wandb.log({"ground truth positive loss": current_loss})

                    if i % 100 == 99: 
                        average_loss.append(running_loss/100)
                        wandb.log({"average overall loss in the last 100 steps": running_loss/100})
                        running_loss = 0
                    #running_loss = 0.0

                if regular_save:
                    if file_idx % 10: # every 10 files: make a snapshot that is permanent
                        torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_model_snapshot_"+str(snapshot_count)+"_fold_"+str(fold))
                        snapshot_count += 1
                    else: # otherwise make a snapshot that is temporary in case of crash
                        torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_model_snapshot_"+str(snapshot_count)+"_fold_"+str(fold))
            
                # run validation after every "file batch"
                intermediate_validation_results = {}
                if clip_nlp_model is not None:
                    validate_classifier(depth_grasp_classifier, val_files, intermediate_validation_results, fold=fold, save_figure=False, results_path=results_path, nlp_prompts=nlp_promts, clip_model=clip_nlp_model, object_type_dict=object_type_dict)
                else:
                    validate_classifier(depth_grasp_classifier, val_files, intermediate_validation_results, fold=fold, save_figure=False, results_path=results_path)
                wandb.log(info_dict_to_wandb_format(intermediate_validation_results))

                precision = intermediate_validation_results["precision"]
                recall = intermediate_validation_results["recall"]
                if precision == 0 and recall == 0:
                    f1_score = 0
                else:
                    f1_score = 2*(precision*recall)/(precision+recall)

                if f1_score > current_best_f1:
                    torch.save(depth_grasp_classifier, results_path+depth_grasp_classifier.name+"_best_model_fold_"+str(fold))
                    current_best_f1 = f1_score

                wandb.log({"current learning rate": scheduler.get_last_lr()[-1]})

            scheduler.step()

        # Redundant, covered by wandb
        # fig, ax = plt.subplots(3)
        # ax[0].plot(loss_vals_gtf)
        # ax[1].plot(loss_vals_gtt)
        # ax[2].plot(average_loss)
        # plt.savefig(PATH[CURRENT_DEVICE]+"training_plot"+str(plot_count)+"_fold_"+str(fold))
        # plot_count += 1
        # if VERBOSE:
        #    plt.show()

        best_model = torch.load(results_path+depth_grasp_classifier.name+"_best_model_fold_"+str(fold))
        tmp_validation_results = {}

        if clip_nlp_model is not None:
            validate_classifier(best_model, val_files, tmp_validation_results, fold=fold, results_path=results_path, nlp_prompts=nlp_promts, clip_model=clip_nlp_model, object_type_dict=object_type_dict)
        else:
            validate_classifier(best_model, val_files, tmp_validation_results, fold=fold, results_path=results_path)
        wandb.log(info_dict_to_wandb_format(intermediate_validation_results)) # last logged model in each fold is the best performing model
        for key in tmp_validation_results.keys():
            dict_list_append(key, tmp_validation_results[key], validation_results)

        wandb.finish()

        if NO_CROSSVAL:
            break

    print("Precision Values: " + str(validation_results["precision"]))
    print("Recall Values: " + str(validation_results["recall"]))
    print("average Precision:" + str(statistics.mean(validation_results["precision"])))
    print("average Recall:" + str(statistics.mean(validation_results["recall"])))

def load_and_eval(model_path=MODEL_PATH[CURRENT_DEVICE], val_file_paths=VAL_FILE_PATHS[CURRENT_DEVICE], results_path=""):
    validation_results = {}
    model = torch.load(model_path)
    validate_classifier(model, val_file_paths, validation_results, results_path=results_path)
    print(validation_results)


def main():
    random.seed(42)
    np.random.seed(seed=42)
    now = datetime.now()
    current_time_and_date = now.strftime("%m.%d.%y_%H:%M:%S")
    results_folder_name = "results_run_" + current_time_and_date
    path = os.path.join(PATH[CURRENT_DEVICE], results_folder_name)
    results_folder_name = results_folder_name + "/"
    os.mkdir(path)

    if TRAIN:
        train_test_depth_pipeline(dataset_path=PATH[CURRENT_DEVICE]+DATA_DIRECTORY[CURRENT_DEVICE], dnt_start=current_time_and_date, results_path=PATH[CURRENT_DEVICE]+results_folder_name, language_prompts_path=LANGUAGE_PROMPTS_PATH)
    else:
        load_and_eval(results_path=PATH[CURRENT_DEVICE]+results_folder_name)

if __name__ == '__main__':
    main()






