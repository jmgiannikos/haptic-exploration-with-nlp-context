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

NUM_EPOCHS = 4
VERBOSE = False
PIXEL_REDUCTION_FACTOR = 2
PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/"
FILE_NAME = "grasp_block_semi_random"
NO_CROSSVAL = True

class Depth_Grasp_Alex_Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# neural network that classifiers wether or not the grasp was successful based on the depth image
class Depth_Grasp_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 2 input channels on first conv layer to accomodate input from both fingers
        self.conv1 = nn.Conv2d(2, 64, kernel_size=11, stride=2, padding=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=2, padding=1)

        self.pool = nn.MaxPool2d(3, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.fc1 = nn.Linear(4608, 4608)
        self.fc2 = nn.Linear(4608, 4608)
        self.fc3 = nn.Linear(4608, 2)
        self.logsoftmax = torch.nn.LogSoftmax(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.avgpool(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.logsoftmax(self.fc3(x))
        return x
    
class depth_dataset(Dataset):
    def __init__(self, data=None, labels=None, data_path=None, transform=None, target_transform=None):

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

def dataset_to_data_loader(dataset=None, data_path="", batch_size=1):
    if dataset is not None:
        dataset = depth_dataset(data=dataset[0], labels=dataset[1])
    else:
        dataset = depth_dataset(data_path=data_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return data_loader

def load_depth_dataset(dataset_path="", normalize=True):
    dataset = dataset=np.load(dataset_path)
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

def validate_classifier(model, val_loader, result_dict):
    pred_labels = []
    j = 0
    for data in val_loader:
        inputs, label = data
        prediction = model(inputs)
        max_pred = -float("inf")
        max_pred_idx = None
        i = 0
        for pred in prediction[0]:
            if pred > max_pred:
                max_pred = pred
                max_pred_idx = i
            i += 1

        if j <= 10:
            render_depth(inputs)
            print("predition:" + str(prediction))
            print("pred label:" + str(max_pred_idx))
            print("gt label:" + str(label))
            j += 1

        pred_labels.append(max_pred_idx)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    for idx, data in enumerate(val_loader, 0):
        _, label = data
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

    if true_positives+false_positives > 0:
        precision = true_positives/(true_positives+false_positives)
    else:
        precision = 0

    if true_positives+false_negatives > 0:
        recall = true_positives/(true_positives+false_negatives)
    else:
        recall = 0

    if "precision" in result_dict.keys():
        result_dict["precision"].append(precision)
    else:
        result_dict["precision"] = [precision]

    if "recall" in result_dict.keys():
        result_dict["recall"].append(recall)
    else:
        result_dict["recall"] = [recall]

    return true_positives, false_positives, true_negatives, false_negatives


def get_dataset_paths(dataset_path_base):
    dataset_paths = []
    i = 1
    while True:
        if i == 1:
            file_suffix = ""    
            path = dataset_path_base + file_suffix + ".npz"

        else:
            file_suffix = str(i)
        
            path = dataset_path_base + file_suffix + ".npz"

        if os.path.isfile(path):
            dataset_paths.append(path)
        else:
            break

        i += 1

    return dataset_paths

def prune_dimensions(array):
    if np.shape(array)[0] == 1:
        return array[0]
    else:
        return array

def train_test_depth_pipeline(dataset_path=""):
    depth_grasp_classifier = Depth_Grasp_Classifier()
    nll_weights = torch.tensor([0.142,1.0])
    criterion = nn.NLLLoss(weight=nll_weights)
    optimizer = optim.SGD(depth_grasp_classifier.parameters(), lr=0.0001)

    dataset_paths = get_dataset_paths(dataset_path)
    #random.shuffle(dataset_paths)

    loss_vals_gtf = []
    loss_vals_gtt = []
    validation_results = {}
    snapshot_count = 0
    for fold, val_file_name in enumerate(dataset_paths): #currently uses exactly one block to evaluate, might want to change to allow multiple blocks later on
        print(f"###### fold: {fold} ######")
        val_data, val_labels = load_depth_dataset(dataset_path=val_file_name)

        val_data = prune_dimensions(val_data)
        val_labels = prune_dimensions(val_labels)

        val_loader = dataset_to_data_loader(dataset=(val_data, val_labels))

        for train_file_name in dataset_paths:
            print(f"--- {train_file_name} ---")
            if train_file_name != val_file_name:
                train_data, train_labels = load_depth_dataset(dataset_path=train_file_name)
                
                train_data = prune_dimensions(train_data)
                train_labels = prune_dimensions(train_labels)

                train_loader = dataset_to_data_loader(dataset=(train_data, train_labels), batch_size=1)

                running_loss = 0
                for epoch in range(NUM_EPOCHS):
                    for i, datapoint in enumerate(train_loader, 0):
                        inputs, label = datapoint

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        outputs = depth_grasp_classifier(inputs)
                        loss = criterion(outputs, label)
                        loss.backward()
                        optimizer.step()

                        # print statistics
                        current_loss = loss.item()
                        #if i % 10 == 9:    # print every 2000 mini-batches
                        if VERBOSE:
                            print(f'[{epoch + 1}, {i + 1:5d}] loss: {current_loss:.3f}')
                        
                        if label.item() == 0:
                            loss_vals_gtf.append(current_loss)
                        else:
                            loss_vals_gtt.append(current_loss)
                        #running_loss = 0.0
            
            torch.save(depth_grasp_classifier, PATH+"_model_snapshot_"+str(snapshot_count))
            snapshot_count += 1

        validate_classifier(depth_grasp_classifier, val_loader, validation_results)
        fig, ax = plt.subplots(2)
        ax[0].plot(loss_vals_gtf)
        ax[1].plot(loss_vals_gtt)
        plt.show()

        if NO_CROSSVAL:
            break

    print("Precision Values: " + str(validation_results["precision"]))
    print("Recall Values: " + str(validation_results["recall"]))
    print("average Precision:" + str(statistics.mean(validation_results["precision"])))
    print("average Recall:" + str(statistics.mean(validation_results["recall"])))

def main():
    train_test_depth_pipeline(PATH + FILE_NAME)

if __name__ == '__main__':
    main()






