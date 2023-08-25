import numpy as np
import os

CURRENT_DEVICE = "drax"
PATH = {"laptop": "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/",
        "drax": "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/"}
DATA_DIRECTORY = {"laptop": "grasp_datasets/",
        "drax": "mixed_object_dataset/"}

def ends_in_npz(filename):
    split_name = filename.split(".")
    return "npz" == split_name[-1]

def get_dataset_paths(dataset_folder_path):
    files_list = os.listdir(dataset_folder_path)
    dataset_paths = [dataset_folder_path + filename for filename in files_list if ends_in_npz(filename)]
    return dataset_paths

def main():
    dataset_paths = get_dataset_paths(PATH[CURRENT_DEVICE]+DATA_DIRECTORY[CURRENT_DEVICE])
    dataset = np.load(dataset_paths[0])
    print(np.shape(dataset["tactile_depth"]))
    print(np.shape(dataset["tactile_color"]))

if __name__ == '__main__':
    main()