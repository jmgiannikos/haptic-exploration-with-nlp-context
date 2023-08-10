import numpy as np
from torch.utils.data import Dataset
import os

def convert_to_class_idx(labels):
    class_indices = []
    for label in labels:
        if label:
            class_indices.append(1)
        else:
            class_indices.append(0)
    return np.asarray(class_indices)

def add_to_global_index_dict(index_dict, start_index, end_offset, file_path):
    for i in range(end_offset):
        index_dict[start_index+i] = (file_path, i)
    return end_offset

class color_dataset(Dataset):
    def __init__(self, data=None, labels=None, object_types=None):
        self.color_labels = convert_to_class_idx(labels)
        self.color_images = data

        self.object_types = object_types

    def __len__(self):
        return len(self.color_labels)

    def __getitem__(self, idx):
        color_image = self.color_images[idx]
        color_label = self.color_labels[idx]

        if self.object_types is not None:
            object_type = self.object_types[idx]
            return color_image, color_label, object_type
        else:
            return color_image, color_label

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