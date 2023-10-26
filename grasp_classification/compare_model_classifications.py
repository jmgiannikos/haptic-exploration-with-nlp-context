import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import os
import random
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay
import wandb
from datetime import datetime
import clip

from grasp_classifiers import *
import grasp_cls_pipeline_configs as configs
import data_processing as dp
import grasp_cls_utils as gcls_utils
import data_loading as dl
import nlp_processing as nlpp

MODEL_FILE_NAMES = {
                    "bottle": {
                        0:"best_model_bottle_fold_0",
                        1:"best_model_bottle_fold_1",
                        2:"best_model_bottle_fold_2",
                        3:"best_model_bottle_fold_3",
                        4:"best_model_bottle_fold_4"
                    },
                    "can": {
                        0:"best_model_can_fold_0",
                        1:"best_model_can_fold_1",
                        2:"best_model_can_fold_2",
                        3:"best_model_can_fold_3",
                        4:"best_model_can_fold_4"
                    },
                    "hair_dryer": {
                        0:"best_model_hair_dryer_fold_0",
                        1:"best_model_hair_dryer_fold_1",
                        2:"best_model_hair_dryer_fold_2",
                        3:"best_model_hair_dryer_fold_3",
                        4:"best_model_hair_dryer_fold_4"
                    },
                    "rubber_duck": {
                        0:"best_model_rubber_duck_fold_0",
                        1:"best_model_rubber_duck_fold_1",
                        2:"best_model_rubber_duck_fold_2",
                        3:"best_model_rubber_duck_fold_3",
                        4:"best_model_rubber_duck_fold_4"
                    },
                    "camera": {
                        0:"best_model_camera_fold_0",
                        1:"best_model_camera_fold_1",
                        2:"best_model_camera_fold_2",
                        3:"best_model_camera_fold_3", 
                        4:"best_model_camera_fold_4"
                    },
                    "microphone": {
                        0:"best_model_microphone_fold_0",
                        1:"best_model_microphone_fold_1",
                        2:"best_model_microphone_fold_2",
                        3:"best_model_microphone_fold_3",
                        4:"best_model_microphone_fold_4"
                    }
} 
MODEL_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/single_object_result_runs/best_models/"
BASELINE_MODEL_FILENAMES = {
    0:"best_model_baseline_fold_0",
    1:"best_model_baseline_fold_1",
    2:"best_model_baseline_fold_2",
    3:"best_model_baseline_fold_3",
    4:"best_model_baseline_fold_4",
}
DATASET_PATHS = {
    "camera": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random7.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random9.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random12.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random4.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random10.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random11.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random5.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random8.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/camera/grasp_camera_semi_random6.npz"]
    },
    "bottle": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random2.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random9.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random8.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random11.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random7.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random6.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random4.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random3.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random10.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/bottle1/grasp_bottle1_semi_random12.npz"]
    },
    "rubber_duck": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random11.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random7.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random4.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random6.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random12.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random9.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random8.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random10.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/rubber_duck/grasp_rubber_duck_semi_random5.npz"]
    },
    "hair_dryer": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random7.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random9.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random5.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random12.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random2.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random6.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random3.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random11.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/hair_drier/grasp_hair_dryer_semi_random8.npz"]
    },
    "can": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random10.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random5.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random3.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random6.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random12.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random2.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random4.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random7.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/can1/grasp_can1_semi_random11.npz"]
    },
    "microphone": {
        0:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random3.npz"],
        1:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random8.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random12.npz"],
        2:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random11.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random7.npz"],
        3:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random9.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random2.npz"],
        4:["/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random4.npz",
           "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/complex_objects/microphone/grasp_microphone1_semi_random10.npz"]
    }
}

FULL_DATASET_RUN = True

def calculate_avg_pairwise_distance(values):
    sum_counter = 0
    running_sum = 0
    for start_idx, value in enumerate(values):
        if start_idx+1 <= len(values):
            for second_idx in range(start_idx+1, len(values)):
                second_value = values[second_idx]
                distance = abs(value - second_value)
                running_sum += distance
                sum_counter += 1
    
    return running_sum/sum_counter

def experiment_avg_pairwise_random(samples=10000):
    running_sum = 0
    for _ in range(samples):
        values =  [random.random(), random.random()]
        running_sum += calculate_avg_pairwise_distance(values)
    
    print(f"for random values avg_pairwise_dist is: {running_sum/samples}")    

def compare_models(fold, dataset_name, result_file):
    dataset_files = DATASET_PATHS[dataset_name][fold]
    model_names = [MODEL_FILE_NAMES[dataset_name][fold],BASELINE_MODEL_FILENAMES[fold]]
    models = {}
    for model_name in model_names:
        models[model_name] = torch.load(MODEL_PATH+model_name)

    gt_labels = []
    model_labels = {}
    model_pos_probabilities = {}

    for val_iter, val_file_name in enumerate(dataset_files):
        ## LOAD DATASET ##
        val_data, val_labels = dl.load_depth_dataset(dataset_path=val_file_name, load_lift_dist=False)

        val_data = dp.prune_dimensions(val_data)
        val_labels = dp.prune_dimensions(val_labels)
        # initialize data loader
        val_loader = dl.haptic_dataset_to_data_loader(dataset=(val_data, val_labels))

        for idx, data in enumerate(val_loader, 0):
            inputs, label = data
            if label.detach().numpy()[0] == 0 or FULL_DATASET_RUN:
                inputs = inputs.to(configs.get_device())
                gt_labels.append(label.detach().numpy()[0])
                for model_name in models.keys():
                    model = models[model_name]
                    model.eval()
                    prediction = model(inputs)
                    prediction = np.exp(prediction.cpu().detach().numpy()[0])
                    predicted_label = np.argmax(prediction)
                    pred_pos_probability = prediction[1]

                    if model_name in model_labels.keys():
                        model_labels[model_name].append(predicted_label)
                    else:
                        model_labels[model_name] = [predicted_label]

                    if model_name in model_pos_probabilities.keys():
                        model_pos_probabilities[model_name].append(pred_pos_probability)
                    else:
                        model_pos_probabilities[model_name] = [pred_pos_probability]

        predictions_equal = []
        avg_pairwise_distances = []
        conf_mats = {}
        for idx, gt_label in enumerate(gt_labels):
            pred_labels = []
            pred_pos_probabilities = []
            all_equal = True
            for model_name in model_labels.keys():
                pred_label = model_labels[model_name][idx]
                if len(pred_labels) != 0:
                    all_equal = all_equal and (pred_label == pred_labels[-1])
                pred_labels.append(pred_label)
                pred_pos_probabilities.append(model_pos_probabilities[model_name][idx])

                if model_name not in conf_mats.keys():
                    conf_mats[model_name] = [0,0,0,0] # tp, fp, fn, tn

                if gt_label == 1:
                    if pred_label == 1:
                        conf_mats[model_name][0] += 1 # true positive
                    else:
                        conf_mats[model_name][2] += 1 # false negative
                else:
                    if pred_label == 0:
                        conf_mats[model_name][3] += 1 # true negative
                    else:
                        conf_mats[model_name][1] += 1 # false positive

            predictions_equal.append(all_equal)
            avg_pairwise_distances.append(calculate_avg_pairwise_distance(pred_labels))

    overall_avg_pairwise_distance = sum(avg_pairwise_distances)/len(avg_pairwise_distances)
    overall_equal_predictions = predictions_equal.count(True)/len(predictions_equal)
    f1_scores = {}

    for model_name in conf_mats.keys():
        tp = conf_mats[model_name][0]
        fp = conf_mats[model_name][1]
        fn = conf_mats[model_name][2]
        if 2*tp+fp+fn == 0:
            f1_scores[model_name] = -1
        else:
            f1_scores[model_name] = (2*tp)/(2*tp+fp+fn)

    result_file.write("### eval results ###\n")
    result_file.write(f"all equal/total samples: {overall_equal_predictions}\n")
    result_file.write(f"average pairwise distance: {overall_avg_pairwise_distance}\n")
    result_file.write(f"f1 scores by model: \n {f1_scores} \n\n")
    result_file.write(f"conf mats by model (tp,fp,fn,tn): \n {conf_mats} \n\n\n")
    assert len(avg_pairwise_distances) == len(predictions_equal)
    return sum(avg_pairwise_distances), predictions_equal.count(True), len(predictions_equal)

def main():
    file = open("/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/model_comp_results.txt", "a")
    pairwise_distance_global_running_sum = 0
    equal_predictions_global_running_sum = 0
    valid_samples_global = 0
    for dataset_name in DATASET_PATHS.keys():
        file.write(f"###### DATASET {dataset_name} ######\n")
        pairwise_distance_local_running_sum = 0
        equal_predictions_local_running_sum = 0
        valid_samples_local = 0
        for fold in range(5):
            file.write(f"### FOLD {fold} ###\n")
            overall_avg_pairwise_distance, overall_equal_predictions, valid_samples = compare_models(fold, dataset_name, file)
            pairwise_distance_local_running_sum += overall_avg_pairwise_distance
            equal_predictions_local_running_sum += overall_equal_predictions
            pairwise_distance_global_running_sum += overall_avg_pairwise_distance
            equal_predictions_global_running_sum += overall_equal_predictions
            valid_samples_local += valid_samples
            valid_samples_global += valid_samples
        model_average_pairwise_distance = pairwise_distance_local_running_sum/valid_samples_local
        model_average_equal_prediction = equal_predictions_local_running_sum/valid_samples_local
        file.write(f"model average all equal/total samples: {model_average_equal_prediction} \n")
        file.write(f"model average pairwise distances: {model_average_pairwise_distance} \n\n\n")
    global_average_pairwise_distance = pairwise_distance_global_running_sum/valid_samples_global
    global_equal_predictions = equal_predictions_global_running_sum/valid_samples_global
    file.write(f"global average all equal/total samples: {global_equal_predictions} \n")
    file.write(f"global average pairwise distances: {global_average_pairwise_distance} \n\n\n")

if __name__ == '__main__':
    main()


