import time
import pybullet as p
import pybullet_data
import pybulletX as px
import tacto
import hydra
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
import urdfpy
import math
import matplotlib.pyplot as plt
import os
import time
import sys

from panda_robot import PandaRobot
from movement_datasets import read_fep_dataset

INCLUDE_GRIPPER = True
DTYPE = 'float64'
SAMPLING_RATE = 1e-3  # 1000Hz sampling rate
FEP_MOVEMENT_DATASET_PATH = "/home/jan-malte/Bachelors Thesis/simulation/movement_datasets/fep_state_to_pid-corrected-torque_55s_dataset.csv"
LIFT_DIST = 0.2
OBJECT_RESET = True
# collection modes:
# 0: random
# 1: semi random
# 2: semi structured
# 3: structured
COLLECTION_MODE = 1
GRID_NODE_NUM = 5
ROT_STEP_WIDTH = 45 # 360 needs to be divisible by this number
GLOBAL_VERBOSE = False
GLOBAL_DATA_DICT = {}
FILE_PATH = "/home/jan-malte/Bachelors Thesis/haptic-exploration-with-nlp-context/simulation/grasp_datasets/"
FILE_NAME = "grasp"
TARGET_ITERATIONS = 50000
REAL_TIME = False
RENDER_DIGITS = False
GLOBAL_TRACE_ITER = None #how regularly we want to collect trace data during move to grasp position and grasp, only works outside of real time mode

def np_arra_dict_getsize(data_dict):
    size = 0
    for key in data_dict.keys():
        size += sys.getsizeof(data_dict[key])
    return size

def render_digits(digits, axsimg=None):
    color, depth = digits.render()

    if axsimg is None:
        fig, axs = plt.subplots(4)
        axsimg = [None, None, None, None]
        axsimg[0] = axs[0].imshow(depth[0], cmap='gist_gray')
        axsimg[1] = axs[1].imshow(depth[1], cmap='gist_gray')
        axsimg[2] = axs[2].imshow(color[0])
        axsimg[3] = axs[3].imshow(color[1])
    else:
        axsimg[0].set_data(depth[0])
        axsimg[1].set_data(depth[1])
        axsimg[2].set_data(color[0])
        axsimg[3].set_data(color[1])

    plt.draw()
    plt.show(block=True)
    return axsimg

def wait_for_resting_object(obj):
    previous_base_pos, _ = obj.get_base_pose()

    # wait 5 time steps
    if not REAL_TIME:
        for _ in range(0,5):
            p.stepSimulation()
    else:
        time.sleep(5*p.getPhysicsEngineParameters()["fixedTimeStep"])

    while True:
        base_pos, _ = obj.get_base_pose()
        if round((previous_base_pos[2] - base_pos[2]),6) == 0: #compare z values
            break
        else:
            previous_base_pos = base_pos

            # wait 5 time steps
            if not REAL_TIME:
                for _ in range(0,5):
                    p.stepSimulation()
            else:
                time.sleep(5*p.getPhysicsEngineParameters()["fixedTimeStep"])



def random_position():
    vec = []
    for _ in range(0,3):
        vec.append(random.uniform(-1, 1))
    return vec

def random_orientation():
    rot_rpy = [180,0,0]
    rot_rpy[2] = random.uniform(0,360)
    rot = R.from_euler("xyz", rot_rpy, degrees=True)
    return rot
    
def calculate_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, minimal_z=0, precise_mesh_grasp=True):
    if isinstance(object_geometry.geometry, urdfpy.Mesh) and precise_mesh_grasp: 
        random_grasp_point = precise_mesh_grasp_point(object_geometry, global_scaling)
    else:
        object_box_width, object_box_depth, object_box_height = calculate_object_box(object_geometry, global_scaling)
        # xyz -> width/depth/height
        # assume origin is in the center
        random_grasp_point = np.random.uniform([-object_box_width/2,-object_box_depth/2,-object_box_height/2], [object_box_width/2, object_box_depth/2, object_box_height/2])   

    grasp_position = globalize_local_grasp(object_base_pos, object_base_ori, random_grasp_point)
    
    #rotation_matrix = rotation.as_matrix()
    #grasp_position = np.matmul(random_grasp_point, rotation_matrix) + object_base_pos

    if grasp_position[2] < minimal_z: #TODO: suboptimal solution -> no longer uniformly distributed -> instead recursion -> slow?
        grasp_position[2] = minimal_z

    return grasp_position

def calculate_grasp_candidate(robot, object_base_pos, object_base_ori, object_geometry, global_scaling, top_grasp=True, precise_mesh_grasp=True, structure_config=None): 
    idx_updates = {}
    if structure_config is None:
        grasp_position = calculate_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, minimal_z=robot.min_z, precise_mesh_grasp=precise_mesh_grasp)
        if top_grasp:
            grasp_orientation = random_orientation()
        else:
            grasp_orientation = robot.valid_orientation(grasp_position)

            while grasp_orientation is None:
                grasp_position = calculate_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, precise_mesh_grasp=precise_mesh_grasp)
                grasp_orientation = robot.valid_orientation(grasp_position)
    else:
        if "orientation" in structure_config.keys() and "position" in structure_config.keys(): #specific orientation and specific position
            grasp_orientation = structured_z_rot(structure_config["orientation"])
            grasp_position = structure_config["position"]
            idx_updates["ori_idx"] = structure_config["orientation"] + ROT_STEP_WIDTH
        elif "orientation" in structure_config.keys() and "pos_idx" in structure_config.keys(): #specific orientation and next valid point on the grid
            grasp_orientation = structured_z_rot(structure_config["orientation"])
            grasp_point_result = calculate_indexed_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, idx=structure_config["pos_idx"], precise_mesh_grasp=precise_mesh_grasp, z_min=robot.min_z)
            grasp_position = grasp_point_result["pt"]
            idx_updates["ori_idx"] = structure_config["orientation"] + ROT_STEP_WIDTH
            idx_updates["pos_idx"] = grasp_point_result["idx"]
        elif "orientation" in structure_config.keys():
            grasp_position = calculate_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, minimal_z=robot.min_z, precise_mesh_grasp=precise_mesh_grasp)
            grasp_orientation = structured_z_rot(structure_config["orientation"])
            idx_updates["ori_idx"] = structure_config["orientation"] + ROT_STEP_WIDTH
 
    return grasp_position, grasp_orientation, idx_updates

def calculate_indexed_grasp_point(object_base_pos, object_base_ori, object_geometry, global_scaling, idx, precise_mesh_grasp=True, z_min=0):
    x_box, y_box, z_box = calculate_object_box(object_geometry, global_scaling)
    trimeshes = object_geometry.geometry.meshes
    trimeshes = [trimesh for trimesh in trimeshes if trimesh.is_watertight]

    lgp_result = calculate_indexed_grasp_point_rec(object_geometry, idx, x_box, y_box, z_box, trimeshes, precise_mesh_grasp)
    idx = lgp_result["idx"]
    local_grasp_point = lgp_result["pt"]

    global_grasp_point = globalize_local_grasp(object_base_pos, object_base_ori, local_grasp_point)

    while True:
        if global_grasp_point[2] >= z_min:
            return {"idx": idx, "pt": global_grasp_point}
        lgp_result = calculate_indexed_grasp_point_rec(object_geometry, idx, x_box, y_box, z_box, trimeshes, precise_mesh_grasp)
        idx = lgp_result["idx"]
        local_grasp_point = lgp_result["pt"]

        global_grasp_point = globalize_local_grasp(object_base_pos, object_base_ori, local_grasp_point)

        # assumes that there is always a valid grasp point in the object. if not this ends up in endless loop

def calculate_indexed_grasp_point_rec(object_geometry, idx, x_box, y_box, z_box, trimeshes, precise_mesh_grasp=True):
    if precise_mesh_grasp and isinstance(object_geometry.geometry, urdfpy.Mesh) and len(trimeshes)>0:
        return generate_local_index_mesh_gp(idx, x_box, y_box, z_box, trimeshes)
    else:
        return generate_local_index_gp(idx, x_box, y_box, z_box)


def generate_local_index_mesh_gp(idx, x_box, y_box, z_box, trimeshes):
    # generate grasp points until one is inside of the object
    result = generate_local_index_gp(idx, x_box, y_box, z_box ) #returns point and index+=1
    idx = result["idx"]
    point = result["pt"]
    
    for trimesh in trimeshes:
        if trimesh.contains(point):
            return result # returns current index and point 
        
    if idx > 10000:
        return None
        
    return generate_local_index_mesh_gp(idx, x_box, y_box, z_box, trimeshes) # recursion with index += 1


def generate_local_index_gp(idx, x_box, y_box, z_box):
    # grasp point in local frame 
    x_idx, y_idx, z_idx = index_extraction(idx)
    x_point = x_idx * (x_box/GRID_NODE_NUM)
    y_point = y_idx * (y_box/GRID_NODE_NUM)
    z_point = z_idx * (z_box/GRID_NODE_NUM)

    idx+=1

    return {"pt": [x_point - x_box/2, y_point - y_box/2, z_point - z_box/2], "idx": idx}


def index_extraction(idx):
    x_idx = idx%GRID_NODE_NUM
    y_idx = math.floor(idx/GRID_NODE_NUM)%GRID_NODE_NUM
    z_idx = math.floor(idx/GRID_NODE_NUM**2)%GRID_NODE_NUM
    return x_idx, y_idx, z_idx

def get_local_rot_mat(object_base_ori, global_grasp_ori):
    # TODO: DOUBLE CHEK IF THIS IS CORRECT MATHEMATICALLY
    inv_rotation = R.inv(R.from_quat(object_base_ori))
    local_rot_mat = np.matmul(inv_rotation.as_matrix(), global_grasp_ori.as_matrix())
    #local_rot = R.from_matrix(local_rot_mat)
    return local_rot_mat

def localize_global_grasp(object_base_pos, object_base_ori, global_grasp_pos):
    # TODO: DOUBLE CHECK IF THIS WORKS AS INTENDED
    inv_rotation = R.inv(R.from_quat(object_base_ori))
    grasp_position = inv_rotation.apply(global_grasp_pos) - object_base_pos
    return grasp_position

def globalize_local_grasp(object_base_pos, object_base_ori, grasp_pos):
    rotation = R.from_quat(object_base_ori)
    grasp_position = rotation.apply(grasp_pos) + object_base_pos
    return grasp_position

def structured_z_rot(angle):
    rot_rpy = [180,0,0]
    rot_rpy[2] = -angle
    rot = R.from_euler("xyz", rot_rpy, degrees=True)
    return rot

def load_geometry_from_cfg(object_cfg):
    urdf_path = px.helper.find_file(object_cfg.urdf_path)
    object = urdfpy.URDF.load(urdf_path)
    object_link = object.base_link
    object_collsision = object_link.collisions
    object_geometry = object_collsision[0].geometry
    return object_geometry

def precise_mesh_grasp_point(object_geometry, global_scaling):
    #object_geometry = load_geometry_from_cfg(object_cfg)
    if isinstance(object_geometry.geometry, urdfpy.Mesh):
        trimeshes = object_geometry.geometry.meshes
    else: 
        return None
    
    trimeshes = [trimesh for trimesh in trimeshes if trimesh.is_watertight] #filter non watertight meshes as the needed calculations do not work here
    if len(trimeshes) > 0:
        volumes = []
        for trimesh in trimeshes:
            volumes.append(trimesh.volume)

        chosen_trimesh = random.choices(trimeshes, weights=volumes)[0]
        bounds = chosen_trimesh.bounds

        proposed_points = np.random.uniform([bounds[0,0],bounds[0,1],bounds[0,2]], [bounds[1,0],bounds[1,1],bounds[1,2]], size=(500,3))

        contained_list = chosen_trimesh.contains(proposed_points)
        i = 0
        for contained in contained_list:
            if contained:
                return global_scaling * proposed_points[i,:]
            i += 1

        return global_scaling * proposed_points[0,:] # default to returning point that is within object box, but not contained inside of trimesh
    else:
        object_box_width, object_box_depth, object_box_height = calculate_object_box(object_geometry, global_scaling)
        # xyz -> width/depth/height
        # assume origin is in the center
        return np.random.uniform([-object_box_width/2,-object_box_depth/2,-object_box_height/2], [object_box_width/2, object_box_depth/2, object_box_height/2])   


def calculate_object_box(object_geometry, global_scaling):
    if isinstance(object_geometry.geometry, urdfpy.Box):
        height = object_geometry.box.size[2]
        width = object_geometry.box.size[0]
        depth = object_geometry.box.size[1]
    elif isinstance(object_geometry.geometry, urdfpy.Cylinder):
        height = object_geometry.cylinder.length
        width = object_geometry.cylinder.radius * 2
        depth = object_geometry.cylinder.radius * 2
    elif isinstance(object_geometry.geometry, urdfpy.Sphere):
        height = object_geometry.sphere.radius * 2
        width = object_geometry.sphere.radius * 2
        depth = object_geometry.sphere.radius * 2
    elif isinstance(object_geometry.geometry, urdfpy.Mesh):
        trimesh_bounds = np.asarray([[-float('inf'), -float('inf'), -float('inf')],
                                     [float('inf'), float('inf'), float('inf')]])
        for trimesh in object_geometry.mesh.meshes:
            # upper and lower x update
            if trimesh.bounds[0,0] > trimesh_bounds[0,0]:
                trimesh_bounds[0,0] = trimesh.bounds[0,0]
            if trimesh.bounds[1,0] < trimesh_bounds[1,0]:
                trimesh_bounds[1,0] = trimesh.bounds[1,0]    

            # upper and lower y update
            if trimesh.bounds[0,1] > trimesh_bounds[0,1]:
                trimesh_bounds[0,1] = trimesh.bounds[0,1]
            if trimesh.bounds[1,1] < trimesh_bounds[1,1]:
                trimesh_bounds[1,1] = trimesh.bounds[1,1]    

            # upper and lower z update
            if trimesh.bounds[0,2] > trimesh_bounds[0,2]:
                trimesh_bounds[0,2] = trimesh.bounds[0,2]
            if trimesh.bounds[1,2] < trimesh_bounds[1,2]:
                trimesh_bounds[1,2] = trimesh.bounds[1,2]    

        width = trimesh_bounds[1,0] - trimesh_bounds[0,0] 
        depth = trimesh_bounds[1,1] - trimesh_bounds[0,1] 
        height = trimesh_bounds[1,2] - trimesh_bounds[0,2]


    else:
        # geometry type could not be resolved
        return None, None, None
    
    return width*global_scaling, depth*global_scaling, height*global_scaling

def check_lift_success(lift_height, resting_z, obj, success_threshhold=0.8):
    current_pos, _ = obj.get_base_pose()
    delta_z = current_pos[2] - resting_z
    if delta_z > success_threshhold * lift_height:
        return True
    else: 
        return False
    
def fit_arrays(data_array, append_array):
    append_shape = np.shape(append_array)
    data_shape = np.shape(data_array)
    for shape_idx in range(1,len(append_shape)):
        diff = append_shape[shape_idx] - data_shape[shape_idx]
        if diff > 0:
            shape = list(np.shape(data_array))
            shape[shape_idx] = diff
            filler_array = np.full(shape=shape, fill_value=np.nan)
            data_array = np.append(data_array, filler_array, shape_idx)
        elif diff < 0:
            shape = list(np.shape(append_array))
            shape[shape_idx] = abs(diff)
            filler_array = np.full(shape=shape, fill_value=np.nan)
            append_array = np.append(append_array, filler_array, shape_idx)
    
    return np.append(data_array, append_array, 0)
  

def add_to_global_data(current_data_dict, global_data_dict):
    if len(global_data_dict.keys()) < len(current_data_dict.keys()): # if the current local data has "undiscovered" keys
        for key in current_data_dict.keys():
                global_data_dict[key] = [current_data_dict[key]]
    else:
        for key in current_data_dict.keys():
                global_data_dict[key] = fit_arrays(global_data_dict[key], [current_data_dict[key]])

def save_global_data_dict(global_data_dict, iterator=1, iterator_max=100, object_name="_object", file_name=FILE_NAME):
    if iterator == 1:
        iter_suffix = ""
    else:
        iter_suffix = str(iterator)

    if COLLECTION_MODE == 0:
        file_name = file_name + object_name + "_random"
    elif COLLECTION_MODE==1:
        file_name = file_name + object_name + "_semi_random"
    elif COLLECTION_MODE==2:
        file_name = file_name + object_name + "_semi_structured"
    elif COLLECTION_MODE==3:
        file_name = file_name + object_name + "_structured"
 
    path = FILE_PATH+file_name+iter_suffix+".npz"

    if os.path.isfile(path):
        if iterator <= iterator_max:
            iterator+=1
            save_global_data_dict(global_data_dict, iterator, object_name=object_name)
        else:
            raise Exception("could not create save file")
    else: 
        np.savez(path, **global_data_dict)

def object_out_of_workspace(object, workspace_bounds):
    object_pos, _ = object.get_base_pose()
    out_of_bounds = False
    workspace_bounds = list(workspace_bounds.values())

    for object_idx in range(0, len(object_pos)-1): # dont check on z axis
        out_of_bounds = out_of_bounds or object_pos[object_idx] > workspace_bounds[object_idx*2]
        out_of_bounds = out_of_bounds or object_pos[object_idx] < workspace_bounds[object_idx*2+1]

    return out_of_bounds

def get_num_saved_data(object_name, file_name=FILE_NAME):

    num_saved_data = 0
    i = 0
    while True:
        if i == 0:
            iter_suffix = ""
        else:
            iter_suffix = str(i+1)

        if COLLECTION_MODE == 0:
            file_name = file_name + object_name + "_random"
        elif COLLECTION_MODE==1:
            file_name = file_name + object_name + "_semi_random"
        elif COLLECTION_MODE==2:
            file_name = file_name + object_name + "_semi_structured"
        elif COLLECTION_MODE==3:
            file_name = file_name + object_name + "_structured"
    
        path = FILE_PATH+file_name+iter_suffix+".npz"

        if os.path.isfile(path):
            dataset = np.load(path)
            num_saved_data += len(dataset["lift_success"])
        else:
            break

        i += 1

    return num_saved_data


@hydra.main(config_name="conf")
def main(cfg):
    objects = {
        "cube": cfg.cube,
        "ico_sphere": cfg.ico_sphere,
        "cylinder": cfg.cylinder,
        "sphere": cfg.sphere,
        "block": cfg.block,
        "cone": cfg.cone,
    }

    global_data_dict = {}

    for key in objects.keys():
        grasp_obj = objects[key]
        """"""
        trajectory = True
        px.init() # initialize pybulletX wrapper

        # Setup plane
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        #p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

        # Setup robot
        panda_robot = PandaRobot(include_gripper=INCLUDE_GRIPPER)

        finger_joint_names = [b'joint_finger_tip_left', b'joint_finger_tip_right']
        digit_links = []
        for idx in range(0,p.getNumJoints(panda_robot.robot_id)):
            if p.getJointInfo(panda_robot.robot_id, idx)[1] in finger_joint_names:
                digit_links.append(idx)

        #setup tacto
        digits = tacto.Sensor(**cfg.tacto)
        
        digits.add_camera(panda_robot.robot_id, digit_links) #doesnt add cameras properly?

        panda_robot.digits = digits

        #setup object
        obj = px.Body(**grasp_obj)
        object_geometry = load_geometry_from_cfg(grasp_obj)
        global_scaling = grasp_obj.global_scaling
        digits.add_body(obj)

        #run pybulletX in diffeerent thread from pybullet
        if REAL_TIME:
            t = px.utils.SimulationThread(real_time_factor=1)
            t.start()
        else:
            panda_robot.global_clock = False

        for joint_idx in panda_robot.joints:
            print(panda_robot.get_joint_info(joint_idx))

        # set friction on robot and object 
        friction_value = 5000
        p.changeDynamics(bodyUniqueId=panda_robot.robot_id, linkIndex=8, lateralFriction=friction_value)
        p.changeDynamics(bodyUniqueId=panda_robot.robot_id, linkIndex=10, lateralFriction=friction_value)
        p.changeDynamics(bodyUniqueId=obj._id, linkIndex=0, lateralFriction=friction_value)

        # initialize axsimg to show digits output
        if RENDER_DIGITS:
            axsimg = render_digits(digits)

        rot_idx = 0
        pos_idx = 0
        iteration = get_num_saved_data(object_name="_" + key)
        start_time = time.time()
        while(iteration < TARGET_ITERATIONS):
            # CONVENTION: all non-trajectory data is saved in a one dimensional array at most, trajectory-data is saved in two dimensional arrays
            current_data_dict = {}
            # tacto reads
            # color, depth = digits.render() # TODO: check how this works, might not be enough queries
            #digits.updateGUI(color, depth)

            panda_robot.reset_state()

            if OBJECT_RESET:
                obj.set_base_pose(grasp_obj.base_position)
            else:    
                wait_for_resting_object(obj)
                if object_out_of_workspace(obj, panda_robot.workspace_bounds): #reset object if it fell out of workspace
                    obj.set_base_pose(grasp_obj.base_position)

            # give simulation a chance to resolve clipping if it occured in reset
            if not REAL_TIME:
                for _ in range(0,5):
                    p.stepSimulation()
            else:
                time.sleep(5*p.getPhysicsEngineParameters()["fixedTimeStep"])

            if RENDER_DIGITS:
                render_digits(digits, axsimg=axsimg)

            # GRASP POINT CALCULATION
            base_pos, base_ori = obj.get_base_pose()
            if COLLECTION_MODE == 1: # random selection of point restricted to within the object, random orientation
                pos, rot, _ = calculate_grasp_candidate(panda_robot, base_pos, base_ori, object_geometry, global_scaling)
            elif COLLECTION_MODE == 2: # random selection of point, then try all claw rotations
                structured_conf = {"orientation": rot_idx%180}
                if rot_idx%180 == 0:
                    pos, rot, idx_conf = calculate_grasp_candidate(panda_robot, base_pos, base_ori, object_geometry, global_scaling, structure_config=structured_conf)
                else:
                    structured_conf["position"] = pos # hold the same position until full rotation was completed
                    pos, rot, idx_conf = calculate_grasp_candidate(panda_robot, base_pos, base_ori, object_geometry, global_scaling, structure_config=structured_conf)
                rot_idx = idx_conf["ori_idx"]
            elif COLLECTION_MODE == 3: # fully structured
                if rot_idx%180 == 0:
                    structured_conf = {"orientation": rot_idx%180, "pos_idx": pos_idx} # move to next position
                    pos, rot, idx_conf = calculate_grasp_candidate(panda_robot, base_pos, base_ori, object_geometry, global_scaling, structure_config=structured_conf)
                    pos_idx = idx_conf["pos_idx"]
                    rot_idx = idx_conf["ori_idx"]
                else:
                    structured_conf = {"orientation": rot_idx%180, "position": pos} # hold position and rotate
                    pos, rot, idx_conf = calculate_grasp_candidate(panda_robot, base_pos, base_ori, object_geometry, global_scaling, structure_config=structured_conf)
                    rot_idx = idx_conf["ori_idx"]

            # MOVEMENT EXECUTION
            object_box_width, object_box_depth, object_box_height = calculate_object_box(object_geometry, global_scaling)
            object_box = [object_box_width, 0, object_box_depth, 0, object_box_height, 0]
            
            if trajectory:
                grasped_pos = panda_robot.grasp_and_lift(pos, rot, obj, object_box, buffer=0.1, distance=LIFT_DIST, verbose=GLOBAL_VERBOSE, data_dict=current_data_dict, target_velocity=0.1, top_grasp=True, trace_iter=GLOBAL_TRACE_ITER, torque_limit=100)
            else:
                #TODO: implement grasp_and_lift_alt to allow non-trajectory grasping
                gripper_pos, success = panda_robot.calculate_inverse_kinematics(pos, rot, verbose=GLOBAL_VERBOSE)

                panda_robot.grasp_at(gripper_pos, max_velocity=1, verbose=GLOBAL_VERBOSE, data_dict=current_data_dict)
                grasped_pos, _ = obj.get_base_pose()
                panda_robot.lift(distance=LIFT_DIST, verbose=GLOBAL_VERBOSE, obj=obj, data_dict=current_data_dict, object_box=object_box)

            # LIFT SUCCESS CHECK
            if grasped_pos is not None:
                lift_success = check_lift_success(lift_height=LIFT_DIST, resting_z=grasped_pos[2], obj=obj)
                
                if GLOBAL_VERBOSE:
                    if lift_success:
                        print("LIFTED SUCCESSFULLY")
                    else:
                        print("LIFT FAILED")

                current_data_dict["lift_success"] = lift_success
                
                size = np_arra_dict_getsize(current_data_dict)
                print(f"the current data dictionary is {size*10e-9}GB big")

                add_to_global_data(current_data_dict, global_data_dict)

                size = np_arra_dict_getsize(global_data_dict)
                print(f"the global data dictionary is {size*10e-9}GB big")

                if size*10e-9 > 2:
                    save_global_data_dict(global_data_dict, object_name="_" + key)
                    print(iteration)
                    global_data_dict = {}



            #TODO: Data saving functionality
            # pre/post object pose (DONE)
            # object-relative grasp pose (local grasp) (DONE, implicit in joint configs)
            # haptic feedback (after grasp) (DONE)
            # lift success (DONE)

            iteration += 1
            panda_robot.reset_state()
        
        stop_time = time.time()
        time_needed = (stop_time - start_time)/60
        total_valid_lifts = len(global_data_dict["lift_success"])
        grasps_per_minute = total_valid_lifts/time_needed 
        print(f"generated {total_valid_lifts} in {time_needed} minutes. Thats {grasps_per_minute} grasps per minute")

        obj_name = "_" + key
        save_global_data_dict(global_data_dict, object_name=obj_name)

if __name__ == '__main__':
    main()
