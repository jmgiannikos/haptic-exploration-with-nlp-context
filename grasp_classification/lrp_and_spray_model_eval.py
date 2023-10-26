import zennit
import torch
import numpy as np
from zennit.image import imgify
import torch.nn as nn
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os

import data_processing as dp
import data_loading as dl
import grasp_cls_pipeline_configs as configs
import nlp_processing as nlpp
from grasp_classifiers import *

RESULT_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/SpRAy results/"
READ_INTERMEDIATE_ATTRIBUTIONS = True
CANONIZER = zennit.canonizers.SequentialMergeBatchNorm()
CUSTOM_LAYER_MAP = [
    (zennit.types.Linear, zennit.rules.AlphaBeta(alpha=2,beta=1, zero_params='bias')),
    (zennit.types.Convolution, zennit.rules.AlphaBeta(alpha=2,beta=1, zero_params='bias')),
    (zennit.types.AvgPool, zennit.rules.Norm()),
    (torch.nn.LogSoftmax, zennit.rules.Pass())
]

COMPOSITES = {
    "epsilon_alpha2_beta1":zennit.composites.EpsilonAlpha2Beta1(canonizers=[CANONIZER]),
    "epsilon_plus":zennit.composites.EpsilonPlus(canonizers=[CANONIZER]),
    "custom with guided backprop":zennit.composites.MixedComposite([
        zennit.composites.GuidedBackprop(),
        zennit.composites.LayerMapComposite(layer_map=CUSTOM_LAYER_MAP)
    ], canonizers=[CANONIZER]),
    "custom without guided backprop": zennit.composites.LayerMapComposite(layer_map=CUSTOM_LAYER_MAP, canonizers=[CANONIZER])
}

BASELINE_MODEL_NAMES = {
    0:"best_model_nlp_cls_fold_0",
    1:"best_model_nlp_cls_fold_1",
    2:"best_model_nlp_cls_fold_2",
    3:"best_model_nlp_cls_fold_3",
    4:"best_model_nlp_cls_fold_4",
}

MODEL_FILE_NAMES = {"bottle": {
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

VERBOSE = False
SHOW_IMGS = 1
SUCCESSFUL_ONLY = True
HEATMAP_MODE = "sums"
NUM_CLUSTERS_BY_MODEL = {}
INPUT_TENSOR_LIST = []

def load_dataset(file_path):
    data, labels = dl.load_depth_dataset(dataset_path=file_path, load_lift_dist=False)
    data = dp.prune_dimensions(data)
    labels = dp.prune_dimensions(labels)
    # initialize data loader
    loader = dl.haptic_dataset_to_data_loader(dataset=(data, labels))

    return loader

def flatten_heatmaps(heatmaps):
    heatmap_shape = np.shape(heatmaps)
    total_dim = 1
    for dim in list(heatmap_shape)[1:]:
        total_dim = total_dim*dim

    heatmaps = np.reshape(heatmaps, (heatmap_shape[0], total_dim)) # flatten heatmaps to fit clustering algorithm

    return heatmaps

def apply_heatmap_mode(heatmaps, heatmap_mode):
    if heatmap_mode == "sums":
        heatmaps_1 = heatmaps[:,:3]
        heatmaps_2 = heatmaps[:,3:]
        heatmaps_1 = np.sum(heatmaps_1, axis=1, keepdims=True)
        heatmaps_2 = np.sum(heatmaps_2, axis=1, keepdims=True)
        heatmaps = np.append(heatmaps_1, heatmaps_2, axis=1)

    return heatmaps

def spray_get_eigenvalues(heatmaps, heatmap_mode="sums"):

    heatmaps = flatten_heatmaps(apply_heatmap_mode(heatmaps, heatmap_mode))

    # fit clustering alg to get affinity matrix
    clustering_alg = SpectralClustering(n_clusters=8, assign_labels='cluster_qr',affinity="nearest_neighbors")
    clustering_alg.fit(heatmaps)
    affinity_matrix = clustering_alg.affinity_matrix_
    affinity_matrix = np.ceil(affinity_matrix.toarray()) # adjust affinity matrix to be in line with spray paper

    # calculate laplacian
    degree_vector = np.sum(affinity_matrix, 1)
    degree_vector = np.asarray(degree_vector).flatten()
    d = np.diag(degree_vector)
    l = d - affinity_matrix

    # get eigenvalues of laplacian
    eigenvalues = np.sort(np.linalg.eig(l)[0])
    plt.scatter(list(range(len(eigenvalues))), eigenvalues)
    plt.show()
    plt.close()

def spectral_clustering(heatmaps, num_clusters, heatmap_mode="sums"):
    flat_heatmaps = flatten_heatmaps(apply_heatmap_mode(heatmaps, heatmap_mode))
    clustering_alg = SpectralClustering(n_clusters=num_clusters, assign_labels='cluster_qr',affinity="nearest_neighbors")
    cluster_labels = list(clustering_alg.fit_predict(flat_heatmaps))
    unique_labels = list(set(cluster_labels))

    label_masks = {}
    for label in unique_labels:
        label_masks[label] = [item==label for item in cluster_labels]

    heatmap_clusters = {}
    for label in label_masks.keys():
        cluster_heatmaps = heatmaps[label_masks[label]]
        heatmap_clusters[label] = cluster_heatmaps

    return heatmap_clusters, label_masks

def store_input_tensor(module, input, output):
    INPUT_TENSOR_LIST.append((type(module), input[0]))
    input[0].retain_grad()

def store_hook(module, input, output):
    # set the current module's attribute 'output' to its tensor
    module.output = output
    # keep the output tensor gradient, even if it is not a leaf-tensor
    output.retain_grad()

def is_valid_layer(layer):
    return isinstance(layer, torch.nn.Conv2d) or isinstance(layer, torch.nn.BatchNorm2d) or isinstance(layer, torch.nn.BatchNorm1d) or isinstance(layer, torch.nn.ReLU) or isinstance(layer, torch.nn.MaxPool2d) or isinstance(layer, torch.nn.Linear) or isinstance(layer, torch.nn.AdaptiveAvgPool2d)

def generate_valid_layer_dict(models, section_name = None):
    if isinstance(models, dict):
        valid_layer_dict = {}
        for section_name in models.keys():
            valid_layer_dict.update(generate_valid_layer_dict(models=models[section_name], section_name=section_name))
    else:
        valid_layer_dict = {}
        for idx, layer in enumerate(models.modules()):
            if is_valid_layer(layer):
                if section_name is None:
                    valid_layer_dict[idx] = layer
                else:
                    valid_layer_dict[f"{section_name}_{idx}"] = layer
    return valid_layer_dict

def lrp(model, file_paths, composite_name="custom with guided backprop", language_prompts=None):        
    model.eval()
    composite = COMPOSITES[composite_name]
    composite.register(model)

    predicted_labels = []
    gt_labels = []
    attributions = None
    inputs = None

    for val_iter, file_path in enumerate(file_paths):

        data_loader = load_dataset(file_path)
        object_type = file_path.split("/")[-1].split("_")[1]
        object_types = [object_type]*len(data_loader)
        
        print(f"loader {val_iter}")
        for idx, data in enumerate(data_loader, 0):
            input, label = data
            input = input.to(configs.get_device())
            input.requires_grad = True

            if language_prompts is not None:
                language_prompt = nlpp.get_language_prompt(object_types[0], language_prompts["val"]) # Hardcoded assuming lrp is conducted on val dataset
                output = model(input, language_prompt)
                #nlp_layer = list(model.nlp_model.modules())[-1]
                #nlp_handle = nlp_layer.register_forward_hook(store_hook)
                first_cls_layer = list(model.classifier.modules())[1]
                first_cls_layer_handle = first_cls_layer.register_forward_hook(store_input_tensor)
            else:
                output = model(input)

            if READ_INTERMEDIATE_ATTRIBUTIONS:
                if language_prompts is None:
                    valid_layer_dict = generate_valid_layer_dict(model)
                else:
                    model_sections = {"classifier": model.classifier, "haptic_cnn": model.cnn_feature_extract, "avg_pool": model.avg_pool}
                    valid_layer_dict = generate_valid_layer_dict(model_sections)

                handles = {}
                for layer_name in valid_layer_dict.keys():
                    layer = valid_layer_dict[layer_name]
                    handles[layer_name] = (layer.register_forward_hook(store_hook))

            class_probability_output = torch.sub(torch.exp(output), torch.tensor((0.5,0.5)).to("cuda")) #torch.clone(output) 
            class_probability_output[0][0] = 0
            
            predicted_label = np.argmax(np.exp(output.cpu().detach().numpy()[0]))
            if (SUCCESSFUL_ONLY and predicted_label == 1) or not SUCCESSFUL_ONLY: 
                grad_outputs = torch.ones_like(output)
                grad_outputs.requires_grad = True
                attribution = torch.autograd.grad(class_probability_output, input, grad_outputs=torch.ones_like(class_probability_output))[0]

                if READ_INTERMEDIATE_ATTRIBUTIONS:
                    intermediate_attributions = []
                    for layer_name in valid_layer_dict.keys():
                        layer_attribution = valid_layer_dict[layer_name].output.grad
                        intermediate_attributions.append((layer_name, layer_attribution))

                if language_prompts is not None:
                    language_attribution = INPUT_TENSOR_LIST[-1].grad
                    language_attribution_sum = np.sum(language_attribution)

                predicted_labels.append(np.argmax(np.exp(output.cpu().detach().numpy()[0])))
                gt_labels.append(label)

                attribution = np.expand_dims(attribution.cpu().detach().numpy(), axis=0)

                if attributions is None:
                    attributions = attribution
                else:
                    attributions = np.append(attributions, attribution, 0)

                input = np.expand_dims(input.cpu().detach().numpy(), axis=0)
                if inputs is None:
                    inputs = input

                else:
                    inputs = np.append(inputs, input, 0)
            elif len(INPUT_TENSOR_LIST) > 0:
                INPUT_TENSOR_LIST.pop() # remove last addition if unnessecary

    return {"attributions": attributions, "inputs": inputs, "predicted_labels": predicted_labels, "gt_labels":gt_labels}

def visualize_lrp_results(attribution, input, predicted_label, gt_label, show_result = False, collapse_strategy = "sum"):
    input_1 = input[:,:3]
    input_2 = input[:,3:]
    attribution_1 = attribution[:,:3]
    attribution_2 = attribution[:,3:]

    if collapse_strategy == "sum":
        attribution_1 = np.sum(attribution_1[0], axis=0, keepdims=True)
        attribution_2 = np.sum(attribution_2[0], axis=0, keepdims=True)
        imgrid = np.concatenate((input_1, input_2), axis=0)
        attrgrid = np.concatenate((attribution_1, attribution_2), axis=0)

    elif collapse_strategy == "spectral":
        input_1r = input_1[:,0]
        input_1g = input_1[:,1]
        input_1b = input_1[:,2]

        input_2r = input_2[:,0]
        input_2g = input_2[:,1]
        input_2b = input_2[:,2]

        attribution_1r = attribution_1[:,0]
        attribution_1g = attribution_1[:,1]
        attribution_1b = attribution_1[:,2]

        attribution_2r = attribution_2[:,0]
        attribution_2g = attribution_2[:,1]
        attribution_2b = attribution_2[:,2]

        imgrid = np.concatenate((input_1r, input_2r, input_1g, input_2g, input_1b, input_2b), axis=0)
        attrgrid = np.concatenate((attribution_1r, attribution_2r, attribution_1g, attribution_2g, attribution_1b, attribution_2b), axis=0)

    if show_result:
        print(f"showing grasp attribution with: \nground truth label: {gt_label} \npredicted label: {predicted_label}")
        display_lrp_results(imgrid=imgrid, attrgrid=attrgrid)
    
    return imgrid, attrgrid

def display_lrp_results(imgrid, attrgrid):
    assert np.shape(imgrid)[0] == np.shape(attrgrid)[0]
    num_images = np.shape(imgrid)[0]

    fig, axs = plt.subplots(nrows=num_images, ncols=2)

    for idx in range(num_images):
        if len(np.shape(imgrid)) > 3:
            image = np.transpose(imgrid[idx], (1,2,0))
        else:
            image = imgrid[idx]

        attr = attrgrid[idx]

        axs[idx,0].imshow(image)
        axs[idx,1].imshow(attr, cmap="seismic", norm=colors.CenteredNorm())

    plt.show(block=True)

def apply_mask(arraylike, mask):
    if not isinstance(arraylike, np.ndarray):
        arraylike = np.asarray(arraylike)
    return arraylike[mask]

def save_lrp_image(imgrid, attrgrid, model, index, predicted_label, gt_label, cluster_label, dataset_name):
    assert np.shape(imgrid)[0] == np.shape(attrgrid)[0]
    num_images = np.shape(imgrid)[0]

    fig, axs = plt.subplots(nrows=num_images, ncols=2)
    fig.suptitle(f"predicted label: {predicted_label}, gt label: {gt_label}")

    for idx in range(num_images):
        if len(np.shape(imgrid)) > 3:
            image = np.transpose(imgrid[idx], (1,2,0))
        else:
            image = imgrid[idx]

        attr = attrgrid[idx]

        axs[idx,0].imshow(image)
        axs[idx,1].imshow(attr, cmap="seismic", norm=colors.CenteredNorm())

    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/")

    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model}/"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model}/")

    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model}/cluster_{cluster_label}/"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model}/cluster_{cluster_label}/")

    plt.savefig(RESULT_PATH+f"dataset_{dataset_name}/model_{model}/cluster_{cluster_label}/image{index}")
    plt.close()


def calculate_average_attribution(heatmaps):
    num_samples = np.shape(heatmaps)[0]
    summed_map = np.sum(heatmaps, axis=0)
    average_map = summed_map/num_samples
    return average_map

def save_average_attribution(average_attribution, model_name, cluster_label, dataset_name):
    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/")

    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model_name}/"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model_name}/")

    if not os.path.isdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model_name}/cluster_{cluster_label}/"):
        os.mkdir(RESULT_PATH+f"dataset_{dataset_name}/model_{model_name}/cluster_{cluster_label}/")

    fig, axs = plt.subplots(nrows=1, ncols=2)

    attribution_1 = average_attribution[:,:3]
    attribution_2 = average_attribution[:,3:]

    attribution_1 = np.sum(attribution_1[0], axis=0, keepdims=True)
    attribution_2 = np.sum(attribution_2[0], axis=0, keepdims=True)

    axs[0].imshow(attribution_1[0], cmap="seismic", norm=colors.CenteredNorm())
    axs[1].imshow(attribution_2[0], cmap="seismic", norm=colors.CenteredNorm())

    plt.savefig(RESULT_PATH+f"dataset_{dataset_name}/model_{model_name}/cluster_{cluster_label}/average_attribution")
    plt.close()

def main():
    train_val_prompts = nlpp.split_nlp_prompt_dict(configs.get_num_folds())
    for dataset_name in DATASET_PATHS.keys():
        for fold in range(5):
            models = {}
            results_by_model = {}
            current_model = MODEL_FILE_NAMES[dataset_name][fold]
            baseline_model = BASELINE_MODEL_NAMES[fold]
            for model_file_name in [current_model, baseline_model]:
                models[model_file_name] = torch.load(MODEL_PATH+model_file_name)

            file_paths = DATASET_PATHS[dataset_name][fold]
            
            for model_name in models.keys():
                model = models[model_name]
                if isinstance(models[model_name],Grasp_Classifier_Raw_CLIP) or isinstance(models[model_name],Depth_Grasp_Classifier_v3_nrm_ltag_col):
                    nlp_prompts = train_val_prompts[fold]
                    results = lrp(model=model, file_paths=file_paths, language_prompts=nlp_prompts)
                else:
                    results = lrp(model=model, file_paths=file_paths)
                results_by_model[model_name] = results

            if VERBOSE:
                for idx in range(SHOW_IMGS):
                    for model_name in results_by_model.keys():
                        results = results_by_model[model_name]
                        print(f"### {model_name} ###")
                        visualize_lrp_results(attribution=results["attributions"][idx], 
                                                                input=results["inputs"][idx], 
                                                                predicted_label=results["predicted_labels"][idx], 
                                                                gt_label=results["gt_labels"][idx],
                                                                show_result=show_img)
            
            for model_name in results_by_model.keys():
                print(f"#### clustering for {model_name} ####")
                results = results_by_model[model_name]

                if model_name not in NUM_CLUSTERS_BY_MODEL.keys() or SUCCESSFUL_ONLY:
                    spray_get_eigenvalues(results["attributions"])
                    num_clusters = int(input("please enter the number of clusters indicated by the eigengap: "))
                    if num_clusters == 0:
                        num_clusters = 1
                else:
                    num_clusters = NUM_CLUSTERS_BY_MODEL[model_name]

                clusters, cluster_masks = spectral_clustering(results["attributions"], num_clusters)

                for label in clusters.keys():
                    print(f"#### cluster with label {label} ####")
                    mask = cluster_masks[label]
                    
                    heatmaps = clusters[label]
                    images = apply_mask(results["inputs"],mask)
                    predicted_labels = apply_mask(results["predicted_labels"],mask)
                    gt_label = apply_mask(results["gt_labels"],mask)

                    show_img = VERBOSE
                    for idx in range(len(predicted_labels)):
                        if idx > SHOW_IMGS:
                            show_img = False
                        imgrid, attrgrid = visualize_lrp_results(heatmaps[idx], images[idx], predicted_labels[idx], gt_label[idx], show_result=show_img)
                        save_lrp_image(imgrid=imgrid, 
                                       attrgrid=attrgrid, 
                                       model=model_name, 
                                       index=idx, 
                                       predicted_label=predicted_labels[idx], 
                                       gt_label=gt_label[idx],
                                       cluster_label=label,
                                       dataset_name=dataset_name)
                        
                    average_attribution = calculate_average_attribution(heatmaps)
                    save_average_attribution(average_attribution, model_name=model_name, cluster_label=label, dataset_name=dataset_name)

if __name__ == '__main__':
    main()
