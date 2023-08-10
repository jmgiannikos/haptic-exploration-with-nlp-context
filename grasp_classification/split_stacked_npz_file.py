import numpy as np

PATH = "/media/jan-malte/17d1286b-1125-41e3-bf20-59faed637169/jan-malte/irl_ico_sphere_dataset/"
FILENAME = "grasp_ico-sphere_6_real.npz"
NUM_CHUNKS = 2

array = np.load(PATH+FILENAME)
for chunk in range(NUM_CHUNKS):
    newfile_dict = {}
    for key in array:
        sub_array = array[key] 
        length = np.shape(sub_array)[0]
        
        start_idx = int(chunk*length/NUM_CHUNKS)
        end_idx = int(chunk*length/NUM_CHUNKS) + int(length/NUM_CHUNKS)
        sub_array = sub_array[start_idx:end_idx]
        print(np.shape(sub_array))
        newfile_dict[key] = sub_array

    np.savez(PATH+FILENAME.split(".")[0]+f"_split{chunk}.npz", **newfile_dict)
