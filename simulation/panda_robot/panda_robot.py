import os
import math
import time
import pybullet as p
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
import scipy as sc
import numpy as np
from sympy import symbols, init_printing, Matrix, eye, sin, cos, pi, lambdify
import matplotlib.pyplot as plt
from functools import partial
import random 

ROBOT_DESCRIPTIONS = [[0, 0, 0, 0, 0, 0.333],
                             [-np.pi/2, 0, 0, 0, 0, 0],
                             [np.pi/2, 0, 0, 0, -0.316, 0],
                             [np.pi/2, 0, 0, 0.0825, 0, 0],
                             [-np.pi/2,  0, 0, -0.0825, 0.384, 0],
                             [np.pi/2, 0, 0, 0, 0, 0],
                             [np.pi/2, 0, 0, 0.088, 0, 0],
                             [0, 0, np.pi/4, 0, 0, 0.265]] # adjusted to account for hand and fingers (0.107 + 0.063 + 0.085 + 0.01) -> (last link, hand, finger, digit allowance)

def R2axisang(R):
        if  (R[0,0] + R[1,1] + R[2,2] - 1)/2 > 1: # edge case, happens due to rounding errors when every R[i,i] is very close to 1
            ang = 0.0
        else:
            ang = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
        Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
        x = (R[2,1] - R[1,2])/Z
        y = (R[0,2] - R[2,0])/Z
        z = (R[1,0] - R[0,1])/Z

        return[x, y, z], ang

def rpyxyz2h(rpy, xyz):
        Ht = [[1,0,0,xyz[0]],
              [0,1,0,xyz[1]],
              [0,0,1,xyz[2]],
              [0,0,0,1]]
        Hx = [[1,0,0,0],
              [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
              [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
              [0,0,0,1]]
        Hy = [[math.cos(rpy[1]),0,math.sin(rpy[1]),0],
              [0,1,0,0],
              [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
              [0,0,0,1]]
        Hz = [[math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
              [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
              [0,0,1,0],
              [0,0,0,1]]
        H = np.matmul(np.matmul(np.matmul(Ht,Hz),Hy),Hx)
        return H

def calculate_forward_kinematics(joint_angles):
    joint_angles = np.asarray(joint_angles)        
    joint_angles = joint_angles.flatten()
    
    Tlink = []
    for i in range(len(ROBOT_DESCRIPTIONS)):
        Tlink.append(rpyxyz2h(ROBOT_DESCRIPTIONS[i][0:3],ROBOT_DESCRIPTIONS[i][3:6]))
    
    Tcurr = [None]*len(ROBOT_DESCRIPTIONS)
    Tjoint = [None]*len(ROBOT_DESCRIPTIONS)
    for i in range(len(joint_angles)):
        Tjoint[i]=[[math.cos(joint_angles[i]),-math.sin(joint_angles[i]),0,0],
                    [math.sin(joint_angles[i]),math.cos(joint_angles[i]),0,0],
                    [0,0,1,0],
                    [0,0,0,1]]
        if i == 0:
            Tcurr[i] = np.matmul(Tlink[i], Tjoint[i])
        else:
            Tcurr[i] = np.matmul(np.matmul(Tcurr[i-1],Tlink[i]),Tjoint[i])

    return Tcurr

def save_to_data_dict(data, key, data_dict):
    if key in data_dict.keys():
            data_dict[key] = np.append(data_dict[key], [data], 0)
    else:
        data_dict[key] = np.asarray([data])

class PandaRobot:
    """"""

    def __init__(self, include_gripper):
        """"""
        self.limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973)
        ]
        
        init_printing(use_unicode=True)

        p.setAdditionalSearchPath(os.path.dirname(__file__) + '/model_description')
        panda_model = "panda_with_gripper.urdf" if include_gripper else "panda.urdf"
        self.robot_id = p.loadURDF(panda_model, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)

        # Set maximum joint velocity. Maximum joint velocity taken from:
        # https://s3-eu-central-1.amazonaws.com/franka-de-uploads/uploads/Datasheet-EN.pdf
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=0, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=1, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=2, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=3, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=4, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=5, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self.robot_id, linkIndex=6, maxJointVelocity=180 * (math.pi / 180))

        # Set DOF according to the fact that either gripper is supplied or not and create often used joint list
        self.dof = p.getNumJoints(self.robot_id) - 1
        self.joints = range(self.dof)

        self.robot_description = ROBOT_DESCRIPTIONS

        self.critical_points_eec = [[-0.04,-0.01,0.015],
                                    [-0.04,0.01,0.015],
                                    [0.04,-0.01,0.015],
                                    [0.04,0.01,0.015],
                                    [0.1,-0.0175,-0.1],
                                    [0.1,0.0175,-0.1],
                                    [0.1,-0.0175,-0.02],
                                    [0.1,0.0175,-0.02],
                                    [-0.1,-0.0175,-0.1],
                                    [-0.1,0.0175,-0.1],
                                    [-0.1,-0.0175,-0.02],
                                    [-0.1,0.0175,-0.02]] # definition of critical points
        
        self.min_z =  np.min(np.asarray(np.abs(self.critical_points_eec))[:,2])

        # values are assumed to be assigned in this exact order. Changing leads to issues.
        self.workspace_bounds = {
            "x_max": 0.6775,
            "x_min": 0.3775,
            "y_max": 0.25,
            "y_min": -0.25,
            "z_max": 0.8,
            "z_min": self.min_z
        }

        self.digits = None
        self.global_clock = True #flag that determines if the robot does its own p.stepSimulation() or if it simply waits. True means global clock is enabled
        self.grasp_obj = None
        # Reset Robot
        self.reset_state()


    def save_data_traces(self, data_dict, *args):
        robot_config = self.get_joint_states()
        save_to_data_dict(robot_config, "robot_config_trace", data_dict)

        color, depth = self.digits.render()
        save_to_data_dict(color, "color_trace", data_dict)
        save_to_data_dict(depth, "depth_trace", data_dict)

        torques = self.read_torques()
        save_to_data_dict(torques, "torque_trace", data_dict)

        if self.grasp_obj is not None:
            object_pose = self.grasp_obj.get_base_pose()
            save_to_data_dict(object_pose, "object_pose_trace", data_dict)

        for arg in args:
            key = arg[0]
            value = arg[1]
            save_to_data_dict(value, key, data_dict)


    def reset_state(self):
        """"""
        target_values = [0,-np.pi/8,0,-np.pi/1.5,0,np.pi/2,np.pi/4,0,0,0,0]
        for j in range(self.dof):
            p.resetJointState(self.robot_id, j, targetValue=target_values[j])
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=target_values)
    def get_joint_states(self):
        joint_states = []
        for joint_idx in range(0,p.getNumJoints(self.robot_id)):
            joint_states.append(p.getJointState(self.robot_id, joint_idx)[0])
        return joint_states

    def get_dof(self):
        """"""
        return self.dof

    def get_joint_info(self, j):
        """"""
        return p.getJointInfo(self.robot_id, j)

    def get_base_position_and_orientation(self):
        """"""
        return p.getBasePositionAndOrientation(self.robot_id)

    def get_position_and_velocity(self):
        """"""
        joint_states = p.getJointStates(self.robot_id, self.joints)
        joint_pos = [state[0] for state in joint_states]
        joint_vel = [state[1] for state in joint_states]
        return joint_pos, joint_vel

    def get_limits(self):
        upper_limits = []
        lower_limits = []
        joint_ranges = []
        for idx in self.joints:
            upper_limit = self.get_joint_info(idx).joint_upper_limit
            lower_limit = self.get_joint_info(idx).joint_lower_limit
            upper_limits.append(upper_limit)
            lower_limits.append(lower_limit)
            joint_ranges.append(upper_limit - lower_limit)
        rest_pos = [0]*self.dof
        return upper_limits, lower_limits, joint_ranges, rest_pos

    def calculate_inverse_kinematics(self, position, orientation, verbose=True, max_iter=200):
        """"""

        joint_config, error, adjustment_mag = self.transpose_jacobian_ik(position, orientation, max_iter=max_iter)
        labels = ["x diff", "y diff", "z diff", "axis_ang_x", "axis_ang_y", "axis_ang_z"]


        success = False
        for idx in range(np.shape(error)[0]):
            if all(list(map(lambda x: x <= 1e-4, error[idx]))):
                success = True
        if verbose:
            fig, axs = plt.subplots(2)
            for idx in range(np.shape(error)[1]):
                axs[0].plot(list(map(lambda x: abs(x), error[:,idx])), label=labels[idx])
            axs[0].legend()
            axs[1].plot(adjustment_mag)

            plt.show()


        joint_config = list(joint_config)+[0.,0.]
        
        return joint_config, success

    def calculate_inverse_dynamics(self, pos, vel, desired_acc):
        """"""
        assert len(pos) == len(vel) and len(vel) == len(desired_acc)
        vector_length = len(pos)

        # If robot set up with gripper, set those positions, velocities and desired accelerations to 0
        if self.dof == 9 and vector_length != 9:
            pos = pos + [0., 0.]
            vel = vel + [0., 0.]
            desired_acc = desired_acc + [0., 0.]

        simulated_torque = list(p.calculateInverseDynamics(self.robot_id, pos, vel, desired_acc))

        # Remove unnecessary simulated torques for gripper if robot set up with gripper
        if self.dof == 9 and vector_length != 9:
            simulated_torque = simulated_torque[:7]
        return simulated_torque

    def set_target_positions(self, desired_pos):
        """"""
        # If robot set up with gripper, set those positions to 0
        if self.dof >= 9:
            desired_pos = desired_pos + [0., 0.]
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=desired_pos)

    def set_torques(self, desired_torque):
        """"""
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.TORQUE_CONTROL,
                                    forces=desired_torque)
        
    def set_target_positions_with_claw(self, desired_pos, gripper_pos1, gripper_pos2):
        desired_pos[8] = gripper_pos1
        desired_pos[10] = gripper_pos2
        p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=desired_pos)
        
    def solve_axis_constraints(self, axis, anchor_point, box_limits):
        min_bounds = []
        max_bounds = []

        # for each dimension solve:
        # d*ax_val + pos[ax] < max_bound 
        # d*ax_val + pos[ax] > min_bound
        i = 0
        for ax_val in axis:
            if ax_val < 0:
                min_bounds.append((box_limits[i] - anchor_point[int(i/2)])/ax_val)
                max_bounds.append((box_limits[i+1] - anchor_point[int(i/2)])/ax_val)
            elif ax_val > 0:
                min_bounds.append((box_limits[i+1] - anchor_point[int(i/2)])/ax_val)
                max_bounds.append((box_limits[i] - anchor_point[int(i/2)])/ax_val)
            else:
                min_bounds.append(-float('inf'))
                max_bounds.append(float('inf')) 

            i += 2
        
        abs_max_bounds = list(map(lambda x: abs(x), max_bounds))
        d_max = max_bounds[abs_max_bounds.index(min(abs_max_bounds))]

        abs_min_bounds = list(map(lambda x: abs(x), min_bounds))
        d_min = min_bounds[abs_min_bounds.index(min(abs_min_bounds))]

        return d_max, d_min 

    def d_top_grasp(self, trajectory_axis, target_pos, workspace_box_limits):
        # d*trajectory_axis[2] + target_pos[2] = max_bound
        # -> max_bound-target_pos[2]/trajectory_axis[2]
        d =  (workspace_box_limits[4]-target_pos[2])/trajectory_axis[2]
        grasp_point = d*trajectory_axis + target_pos

        # ensure trajectory start point is within bounds
        valid_d = grasp_point[0] >= workspace_box_limits[1] and grasp_point[0] <= workspace_box_limits[0]
        valid_d = valid_d and grasp_point[1] >= workspace_box_limits[3] and grasp_point[1] <= workspace_box_limits[2]
        valid_d = valid_d and grasp_point[2] <= workspace_box_limits[4]

        if valid_d:
            return d
        else:
            return None

    def grasp_with_trajectory(self, target_pos, target_ori, object_pos, object_ori, object_box, buffer, max_velocity=1, torque_limit=50, max_iter = 2400, verbose=True, target_velocity=0.1, data_dict=None, top_grasp=False, trace_iter=None):
        """
        object_box is assumed to be in the format [x_max, x_min, y_max, y_min, z_max, z_min] in local frame coordinates
        """
        
        # calculate staging position
        trajectory_axis = target_ori.apply([0,0,1]) # z axis of manipulator serves as approach axis.
        
        workspace_box_limits = [self.workspace_bounds["x_max"], 
                                self.workspace_bounds["x_min"], 
                                self.workspace_bounds["y_max"],
                                self.workspace_bounds["y_min"],
                                self.workspace_bounds["z_max"],
                                self.workspace_bounds["z_min"]]
        
        if top_grasp:
            workspace_box_limits[4] = workspace_box_limits[4] * 0.6
            d = self.d_top_grasp(trajectory_axis, target_pos, workspace_box_limits)
            if d is None:
                return False
        else:
        
            d_max_outer, d_min_outer = self.solve_axis_constraints(trajectory_axis, target_pos, workspace_box_limits)
            
            #transform into object box frame
            box_frame_axis = object_ori.apply(trajectory_axis)
            box_frame_target_pos = object_ori.apply(target_pos-object_pos) 

            d_max_inner, d_min_inner = self.solve_axis_constraints(box_frame_axis, box_frame_target_pos, object_box) 

            valid_d = []

            if d_max_inner + buffer <= d_max_outer:
                valid_d.append(d_max_inner + buffer)
            if d_min_inner - buffer >= d_min_outer:
                valid_d.append(d_min_inner - buffer)

            if len(valid_d) == 0:
                return False
            elif len(valid_d) == 1:
                d = valid_d[0]
            elif len(valid_d) == 2:
                d = None
                staging_pt_dist = float('inf')
                
                current_joint_config, _ = self.get_position_and_velocity()
                gripper_pose = calculate_forward_kinematics(current_joint_config[0:8])[-1] # only the first 7 joints are relevant
                gripper_pos = gripper_pose[3,0:3]

                for tmp_d in valid_d:
                    staging_pt = tmp_d * trajectory_axis + target_pos
                    if np.linalg.norm(staging_pt - gripper_pos) < staging_pt_dist:
                        d = tmp_d

        # move along trajectory (10 steps)
        step_width = d/10
        local_verbose = verbose
        ik_max_iter = 200
        mov_max_iter = max_iter
        for i in range(0,11):
            current_gripper_goal = (d - step_width * i)* trajectory_axis + target_pos
            gripper_pos, success = self.calculate_inverse_kinematics(current_gripper_goal, target_ori, verbose=local_verbose, max_iter=ik_max_iter)
            break_torque = self.move_to_joint_config(gripper_pos, max_iter=mov_max_iter, max_velocity=max_velocity, verbose=verbose, data_dict=data_dict, collision_avoidance=True, trace_iter=trace_iter)
            # if collision was detected stop executing grasp
            if break_torque:
                break        
            # change parameters for trajectory (only small spatial position updates)
            local_verbose = False
            ik_max_iter = 20
            mov_max_iter = 240

        self.grasp(verbose=verbose, target_velocity=target_velocity, torque_limit=torque_limit, data_dict=data_dict, trace_iter=trace_iter)
        return True

    def read_torques(self):
        try:
            joint_states = p.getJointStates(bodyUniqueId=self.robot_id,
                        jointIndices=range(0,8))
            torques = joint_states.applied_joint_motor_torque
        except:
            for joint_idx in range(0,8):
                p.enableJointForceTorqueSensor(bodyUniqueId=self.robot_id,
                                        jointIndex=joint_idx)
            torques = p.getJointStates(bodyUniqueId=self.robot_id,
                        jointIndices=range(0,8)).applied_joint_motor_torque
            
        return torques

    def check_torques(self, previous_torques=None, torque_limit=60):
        torques = self.read_torques()
            
        if previous_torques is not None:
            all_torques = np.append(previous_torques, [torques], 0)
            abs_torques = np.absolute(all_torques)   
        else:
            abs_torques = np.absolute(torques)

        torques_metric = np.average(abs_torques)
        torques = np.append(torques, [torques_metric], 0)

        return torques, torques_metric > torque_limit

    def grasp(self, verbose=True, target_velocity=0.1, max_iter=2400, torque_limit=0, data_dict=None, trace_iter=None):
        if target_velocity is None:
            # doesnt work very well. Fails to lift 
            joint_idxs = [8,10] 
            for idx in range(0,2):
                joint_idx = joint_idxs[idx]
                p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=joint_idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=0.04,
                                        maxVelocity=0.2)
        
        else:
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=[8,10],
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocities=[target_velocity,target_velocity])

        
        grip_torques_8 = []
        grip_torques_10 = []

        iter = 0
        while True:
            # step time
            if self.global_clock:
                time.sleep(p.getPhysicsEngineParameters()["fixedTimeStep"])
            else:
                if trace_iter is not None and iter%trace_iter==0:
                    self.save_data_traces(data_dict)
                p.stepSimulation()

            try:
                joint_states = p.getJointStates(bodyUniqueId=self.robot_id,
                            jointIndices=[8,10])
                torque = joint_states.applied_joint_motor_torque
            except:
                p.enableJointForceTorqueSensor(bodyUniqueId=self.robot_id,
                                            jointIndex=8)
                p.enableJointForceTorqueSensor(bodyUniqueId=self.robot_id,
                                            jointIndex=10)
                torque = p.getJointStates(bodyUniqueId=self.robot_id,
                            jointIndices=[8,10]).applied_joint_motor_torque
                
            # positional reset workaround.
            if joint_states.joint_position[0] > 0.04:
                p.resetJointState(self.robot_id, 8, targetValue=0.04)
            elif joint_states.joint_position[0] < 0:
                p.resetJointState(self.robot_id, 8, targetValue=0)

            # positional reset workaround.
            if joint_states.joint_position[1] > 0.04:
                p.resetJointState(self.robot_id, 10, targetValue=0.04)  
            elif joint_states.joint_position[1] < 0:
                p.resetJointState(self.robot_id, 10, targetValue=0)

            # end of 
            if all(map(lambda val: abs(val) > torque_limit, torque)):
                break
            if iter > max_iter:
                if verbose:
                    print("WARNING: robot did not grasp with full strenght")
                break

            grip_torques_8.append(torque[0])
            grip_torques_10.append(torque[1])
            iter += 1

        # collect contact data
        if data_dict is not None:
            color, depth = self.digits.render()
            data_dict["tactile_color"] = np.asarray(color)
            data_dict["tactile_depth"] = np.asarray(depth)
        
        if verbose:
            if len(grip_torques_8) > 0:
                print("max torque 8:" + str(max(grip_torques_8)))
                print("min torque 8:" + str(min(grip_torques_8)))
            if len(grip_torques_10) > 0:
                print("max torque 10:" + str(max(grip_torques_10)))
                print("min torque 10:" + str(min(grip_torques_10)))

            #print(grip_torques_10)
            fig, axs = plt.subplots(1)
            axs.plot(range(len(grip_torques_8)), grip_torques_8, label="grip torque 8")
            axs.plot(range(len(grip_torques_10)), grip_torques_10, label="grip torque 10")
            axs.legend()

            plt.show()

    def move_to_joint_config(self, desired_config, max_velocity=1, max_iter=2400, grasping=False, verbose=True, data_dict=None, obj=None, obj_box=None, collision_avoidance=False, trace_iter=None):
        for joint_idx in self.joints[0:8]:
            p.setJointMotorControl2(bodyIndex=self.robot_id,
                                        jointIndex=joint_idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=desired_config[joint_idx],
                                        maxVelocity=max_velocity)

        if self.global_clock:
            time.sleep(p.getPhysicsEngineParameters()["fixedTimeStep"])
        else:
            p.stepSimulation()

        iter = 0
        while True:
            # read velocities
            velocity = p.getJointStates(bodyUniqueId=self.robot_id,
                           jointIndices=self.joints[0:8]).joint_velocity

            # reset gripper positions regularly if not grasping
            if not grasping:
                p.resetJointState(self.robot_id, 8, targetValue=0)
                p.resetJointState(self.robot_id, 10, targetValue=0)

            # check for no further movement
            if all(map(lambda val: round(val,5) == 0 ,velocity)):
                break

            # check for maximum iterations reached
            if iter > max_iter:
                if verbose:
                    print("WARNING: robot did not reach resting position")
                break
            if collision_avoidance:
                # collect current torques
                if iter == 0:
                    tmp_torques = self.read_torques()
                    #tmp_torques, break_torque = self.check_torques(tmp_torques)
                    torques = [np.append(tmp_torques, [0])] 
                elif iter < 5:
                    tmp_torques = self.read_torques()
                    torques = np.append(torques, [np.append(tmp_torques, [0])], 0)
                else:
                    tmp_torques, break_torque = self.check_torques(torques[-5:-1,0:8])
                    torques = np.append(torques, [tmp_torques], 0)

                    # break if any joint torque is too high
                    if break_torque:
                        # stop movement by moving to current position
                        current_joint_config = self.get_joint_states()
                        p.setJointMotorControlArray(bodyIndex=self.robot_id,
                                        jointIndices=range(0,8),
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=current_joint_config[0:8])
                        break
            
            # step simulation
            if self.global_clock:
                time.sleep(p.getPhysicsEngineParameters()["fixedTimeStep"])
            else:
                if trace_iter is not None and iter%trace_iter==0:
                    self.save_data_traces(data_dict)

                p.stepSimulation()

            iter += 1

        # SHOW TORQUE GRAPH
        if verbose and collision_avoidance:
            torques = np.asarray(torques)
            for joint_idx in range(0,8):
                print("max torque " + str(joint_idx) + ":" + str(max(torques[:,joint_idx])))
                print("min torque " + str(joint_idx) + ":" + str(min(torques[:,joint_idx])))

            #print(grip_torques_10)
            fig, axs = plt.subplots(2)
            for joint_idx in range(0,8):
                axs[0].plot(range(len(torques[:,joint_idx])), torques[:,joint_idx], label="grip torque " + str(joint_idx))
                axs[0].legend()

            axs[1].plot(range(len(torques[:,-1])), torques[:,-1], label="average grip torque")

            plt.show()

        # DATA COLLECTION
        if data_dict is not None:
            current_joint_config = self.get_joint_states()
            if "joint_configs" not in data_dict.keys():
                data_dict["joint_configs"] = np.asarray([current_joint_config])
            else:
                data_dict["joint_configs"] = np.append(data_dict["joint_configs"], [current_joint_config], 0)

            if obj is not None and obj_box is not None:
                base_pos, base_ori = obj.get_base_pose()
                object_pos = base_pos + np.asarray([-obj_box[0]/2,-obj_box[2]/2,-obj_box[4]/2]) #correct base pos to the lower left corner
                object_ori = R.from_quat(base_ori)
            
                if "object_positions" not in data_dict.keys():
                    data_dict["object_positions"] = np.asarray([object_pos])
                else:
                    data_dict["object_positions"] = np.append(data_dict["object_positions"], [object_pos], 0)

                if "object_orientations" not in data_dict.keys():
                    data_dict["object_orientations"] = np.asarray([object_ori])
                else:
                    data_dict["object_orientations"] = np.append(data_dict["object_orientations"], [object_ori], 0)
        if collision_avoidance:
            return break_torque # return wether or not the movement was stopped through collision
        
    def grasp_at(self, desired_pos, max_velocity=1, torque_limit=100, max_iter = 2400, verbose=True, target_velocity=0.1, data_dict=None):

        self.move_to_joint_config(desired_config=desired_pos, max_velocity=max_velocity, max_iter=max_iter, verbose=verbose, data_dict=data_dict)

        self.grasp(torque_limit=torque_limit, target_velocity=target_velocity, max_iter=max_iter, verbose=verbose, data_dict=data_dict)


    def lift(self, distance=0.2, max_velocity=1, max_iter = 2400, verbose=True, data_dict=None, obj=None, object_box=None):
        current_joint_config, _ = self.get_position_and_velocity()
        gripper_pose = calculate_forward_kinematics(current_joint_config[0:8])[-1]
        current_position = gripper_pose[0:3,3]
        current_orientation = R.from_matrix(gripper_pose[0:3,0:3])

        target_position = current_position + np.asarray([0,0,distance]) #add to z to lift up 

        target_joint_config, success = self.calculate_inverse_kinematics(target_position, current_orientation, False)
        self.move_to_joint_config(desired_config=target_joint_config, max_velocity=max_velocity, max_iter=max_iter, grasping=True, verbose=verbose)
        
        if data_dict is not None and obj is not None and object_box is not None: # only query data collection if all nessecary parameters are set
            base_pos, base_ori = obj.get_base_pose()
            object_pos = base_pos + np.asarray([-object_box[0]/2,-object_box[2]/2,-object_box[4]/2]) #correct base pos to the lower left corner
            object_ori = R.from_quat(base_ori)
            
            data_dict["final_pos"] = object_pos
            data_dict["final_ori"] = object_ori.as_quat()


    def grasp_and_lift(self, desired_pos, desired_orientation, obj, object_box, buffer=0.1, max_velocity=1, max_iter = 2400, distance=0.2, torque_limit=100, verbose=True, target_velocity=0.1, data_dict=None, top_grasp=False, trace_iter=None):
        base_pos, base_ori = obj.get_base_pose()
        object_pos = base_pos + np.asarray([-object_box[0]/2,-object_box[2]/2,-object_box[4]/2]) #correct base pos to the lower left corner
        object_ori = R.from_quat(base_ori) 

        # save initial pose into data dict
        data_dict["initial_pos"] = object_pos
        data_dict["initial_ori"] = object_ori.as_quat()
        
        success = self.grasp_with_trajectory(desired_pos, desired_orientation, object_pos, object_ori, object_box, buffer, max_velocity=max_velocity, max_iter=max_iter, torque_limit=torque_limit, verbose=verbose, target_velocity=target_velocity, data_dict=data_dict, top_grasp=top_grasp, trace_iter=trace_iter)
        grasped_position, grasped_ori = obj.get_base_pose()

        # save after-grasp pose into data dict
        data_dict["after_grasp_pos"] = grasped_position
        data_dict["after_grasp_ori"] = grasped_ori

        if success:
            self.lift(distance=distance, max_velocity=max_velocity, max_iter=max_iter, obj=obj, object_box=object_box, data_dict=data_dict, verbose=verbose)
            return grasped_position
        else:
            return None

    def grid_move(self):
        x_range = range(0,20)
        y_range = range(0,20)
        for x in x_range:
            for y in y_range:
                desired_pos = list(self.calculate_inverse_kinematics([x/10, y/10, 0.02], [1,0,0,0])) + [0,0]
                p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                        jointIndices=self.joints,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=desired_pos)
        
    def test_sequence(self):
        for index in self.joints:
            print(index)
            target_vels = [0]*self.dof
            target_vels[index] = -0.5
            p.setJointMotorControlArray(bodyUniqueId=self.robot_id,
                                    jointIndices=self.joints,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocities = target_vels)
            self.reset_state()
    
    def transpose_jacobian_ik(self, position, rotation, atol=1e-4, max_iter=200):
        goal_rot_mat = rotation.as_matrix()
        

        q_init, _ = self.get_position_and_velocity()
        q_init = np.asarray(q_init[0:8]).reshape(8, 1) # len 8 even with 7 dof because the last link has to be accounted for TODO: change if this causes issues

        delta_theta = np.asarray([0]*8)
        adjustment_mag= []

        iteration = 0
        lowest_error = float('inf')
        best_solution = None
        while True:
            #calculate the current joint angles by adding the delta theta to the initial angles
            current_joint_angles = q_init.T + delta_theta

            #calculate the current ee pose (last pose in the forwards kinematics chain)
            current_ee_pose_matrix = calculate_forward_kinematics(current_joint_angles)[-1]

            #difference between current ee pose and target ee pose
            axis_angle_rot_error = self.calculate_axis_angle_error(goal_rot_mat, current_ee_pose_matrix[0:3,0:3])
            positional_error = position - current_ee_pose_matrix[0:3,3]
            delta_ee_pose = np.append(positional_error, axis_angle_rot_error, 0)

            if iteration == 0:
                errors = [delta_ee_pose]
            else:
                errors = np.append(errors, [delta_ee_pose], 0)
            
            #calculate current jacobian matrix
            current_jacobian = self.calculate_jacobian(current_joint_angles)

            # calculate alpha as in: http://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf
            alpha = self.calculate_alpha(current_jacobian, delta_ee_pose)

            # transpose jacobian method
            delta_theta_diff = alpha * np.dot(current_jacobian.T, delta_ee_pose)

            # update and format delta theta (the difference in joint angles between starting config and goal config)
            delta_theta = delta_theta + delta_theta_diff
            delta_theta = delta_theta.flatten()
            
            adjustment_mag.append(np.linalg.norm(delta_theta_diff))

            if np.max(np.abs(delta_ee_pose)) < lowest_error:
                best_solution = delta_theta
                lowest_error = np.max(np.abs(delta_ee_pose))


            if np.max(np.abs(delta_ee_pose)) <= atol:
                joint_config = (current_joint_angles).flatten().tolist()
                if joint_config[-1] != 0:
                    print("WARNING: joint config of fixed joint not 0")
                joint_config = joint_config[0:-1]
                return joint_config, errors, adjustment_mag
            elif iteration > max_iter:
                joint_config = (q_init.T + best_solution).flatten().tolist()
                if joint_config[-1] != 0:
                    print("WARNING: joint config of fixed joint not 0")
                joint_config = joint_config[0:-1]
                return joint_config, errors, adjustment_mag

            iteration += 1

    def calculate_axis_angle_error(self, goal_rot_mat, curr_rot_mat):
        ori_error_mat = np.matmul(goal_rot_mat, curr_rot_mat.T)
        axis, angle = R2axisang(ori_error_mat)
        if angle > 0.1:
            angle = 0.1
        elif angle <-0.1:
            angle = -0.1

        return [axis[0]*angle, axis[1]*angle, axis[2]*angle]

    def calculate_alpha(self, current_jacobian, e):
        jxjt = np.dot(current_jacobian, current_jacobian.T)
        jxjtxe = np.dot(jxjt, e)
        exjxjtxe = np.dot(e.T,jxjtxe)
        jxjtxexjxjtxe = np.dot(jxjtxe.T, jxjtxe)
        mat_alpha = exjxjtxe/jxjtxexjxjtxe
        return mat_alpha
    
    def get_current_pose(self):
        q_init, _ = self.get_position_and_velocity()
        fk_pose = self.A_lamb(q_init[0], q_init[1], q_init[2], q_init[3], q_init[4], q_init[5], q_init[6])
        pos = fk_pose[9:].flatten()
        ori_mat = R.from_matrix(fk_pose[0:9].reshape(3,3).T)
        ori = ori_mat.as_quat()
        return pos, ori, q_init
    
    def calculate_jacobian(self, joint_angles):
        Tcurr = calculate_forward_kinematics(joint_angles)
        J = np.zeros((6,len(Tcurr)))
        for i in range(len(Tcurr)-1):
            p=Tcurr[-1][0:3,3]-Tcurr[i][0:3,3]
            a=Tcurr[i][0:3,2]
            J[0:3,i]=np.cross(a,p)
            J[3:6,i]=a # deviation from skript code: a is only 3 long and cannot fit into a slice of size 4
        return J
    
    def valid_orientation(self, grasp_pos, allowance=0, r_limits=(-90,90), p_limits=(90,270), y_limits=(-90,90)): # default allows only top grasps
        height_gc = grasp_pos[2]
        r_vals = list(range(r_limits[0],r_limits[1]))
        p_vals = list(range(p_limits[0],p_limits[1]))

        random.shuffle(r_vals)
        random.shuffle(p_vals)

        for r in r_vals:
            for p in p_vals:
                rotation = R.from_euler("xyz", [r,p-90,random.uniform(y_limits[0],y_limits[1])],degrees=True) # y value should have no impact on z elevation since it is a rotation around the global z axis
                critical_points_gc = rotation.apply(self.critical_points_eec)
                z_vals = critical_points_gc[:,2].flatten()
                if all(map(lambda x: x >= allowance, map(lambda val: val+height_gc, z_vals))):
                    return rotation
        return None
        
        


        



            