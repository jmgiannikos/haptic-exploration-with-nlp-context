from sklearn.model_selection import StratifiedKFold
import numpy as np

def dict_list_append(key, val, target_dict):
    if key in target_dict.keys():
        target_dict[key].append(val)
    else:
        target_dict[key] = [val]

def ends_in_npz(filename):
    split_name = filename.split(".")
    return "npz" == split_name[-1]

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

def get_crossval_idxs(data, labels):
    splitter = StratifiedKFold()
    splits = splitter.split(X=data, y=labels)
    return splits 

def array_append(value, array=None, axis=0):
    if array is None:
        array = np.array([value])
    else:
        array = np.append(array, np.array([value]), axis=axis)
    return array