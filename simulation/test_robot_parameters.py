import time
import pybullet as p
import pybullet_data
import pybulletX as px
import tacto
import hydra
import random

from panda_robot import PandaRobot
from movement_datasets import read_fep_dataset

INCLUDE_GRIPPER = True
DTYPE = 'float64'
SAMPLING_RATE = 1e-3  # 1000Hz sampling rate
FEP_MOVEMENT_DATASET_PATH = "/home/jan-malte/Bachelors Thesis/simulation/movement_datasets/fep_state_to_pid-corrected-torque_55s_dataset.csv"

def random_position():
    vec = []
    for _ in range(0,3):
        vec.append(random.uniform(0.0, 8.0))
    return vec

def random_orientation():
    vec = random_position()
    vec.append(0)
    return vec

@hydra.main(config_name="conf")
def main(cfg):
    """"""
    px.init() # initialize pybulletX wrapper

    # Setup plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")

    #p.resetDebugVisualizerCamera(**cfg.pybullet_camera)

    # Setup robot
    panda_robot = PandaRobot(include_gripper=INCLUDE_GRIPPER)

    finger_joint_names = [b'panda_finger_joint1', b'panda_finger_joint2']
    digit_links = []
    for idx in range(0,p.getNumJoints(panda_robot.robot_id)):
        if p.getJointInfo(panda_robot.robot_id, idx)[1] in finger_joint_names:
            digit_links.append(idx)

    #setup tacto
    digits = tacto.Sensor(**cfg.tacto)
    
    digits.add_camera(panda_robot.robot_id, digit_links) #doesnt add cameras properly

    print(digits.cameras)
    print(digits.nb_cam)

    #setup object
    obj = px.Body(**cfg.object1)
    digits.add_body(obj)

    #run pybulletX in diffeerent thread from pybullet
    t = px.utils.SimulationThread(real_time_factor=1.0)
    t.start()

    idx = 0
    claw_positions = range(0,20)
    while(True):
        # tacto reads
        color, depth = digits.render()
        digits.updateGUI(color, depth)

        #pos = random_position()
        #ori = random_orientation()
        #gripper_pos = list(panda_robot.calculate_inverse_kinematics(pos, ori))
        gripper_pos, _ = panda_robot.get_position_and_velocity()
        print(str(gripper_pos[-2]) + ", " + str(gripper_pos[-1]))
        gripper_pos = gripper_pos[0:-2]

        time.sleep(0.4)

if __name__ == '__main__':
    main()