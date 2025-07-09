import time
import math
import torch
import numpy as np
import pybullet as p
import numpy.linalg as lin

import utils.Lie as lie
import utils.SE3_functions as se3
import utils.franka_functions as ff
from models import load_pretrained


class PandaRobot:
    def __init__(self, enable_gui=True, enable_realtime=True, holder_move=None, position=[0,0,0], holder=True, device="cuda:0"):
        # GUI setting
        self.enable_gui = enable_gui
        if self.enable_gui:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        if enable_realtime:
            p.setRealTimeSimulation(1)
        
        self.holder = holder
        
        # environment setting
        self._plane_id = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.055])
        if self.holder:
            self._holder_init_position = [0.4695, -0.4385, -0.055]
            self._holder_init_eulerangle = [math.pi/2, 0, 0.1383]
            self._holder_init_orientation = p.getQuaternionFromEuler(self._holder_init_eulerangle)
        self._motion = torch.eye(4).to(torch.float32)
        
        if holder_move is not None:
            self._motion = holder_move # (4,4)
            holder_T = np.eye(4)
            holder_T[:3,3] = np.expand_dims(np.array(self._holder_init_position), axis=0)
            holder_T[:3,:3] = lie.quaternion_to_SO3(torch.tensor(list(self._holder_init_orientation)).unsqueeze(0)).squeeze().numpy()
            self._holder_init_T = self._motion @ holder_T
            self._holder_init_position = list(self._holder_init_T[:3,3])
            self._holder_init_orientation = lie.SO3_to_quaternion(torch.tensor(self._holder_init_T[:3,:3]).unsqueeze(0)).squeeze().numpy()
            self._motion = torch.tensor(self._motion).to(torch.float32)
            
            
        if self.holder:
            self._umbrella_holder_id = p.loadURDF('assets/umbrella_holder/umbrella_holder.urdf',
                                                self._holder_init_position,
                                                self._holder_init_orientation,
                                                useFixedBase=True)
            self._umbrella_holder_pos, self._umbrella_holder_ori = p.getBasePositionAndOrientation(self._umbrella_holder_id)
        
        # robot setting
        self._robot_body_id = p.loadURDF("assets/franka_panda/panda_umbrella.urdf",
                                         position,
                                         p.getQuaternionFromEuler([0, 0, 0]),
                                         useFixedBase=True,
                                         flags=p.URDF_USE_SELF_COLLISION)
        
        # get grasp point index
        num_joints = p.getNumJoints(self._robot_body_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self._robot_body_id, i)
            joint_name = joint_info[1].decode('utf-8')
            if joint_name == "panda_grasptarget_hand":
                self._robot_grasptarget_idx = i
            if joint_name == "grasptarget_umbrella":
                self._grasp_joint_idx = i
                break
        
        # background setting
        self._sampling_num = 240
        self._sampling_rate = 1/self._sampling_num
        p.setTimeStep(self._sampling_rate)
        p.setGravity(0, 0, -9.8)
        
        # joint limit
        # joint 0~6 : revolute // 7 : fixed // 8,9 : gripper
        # link 0~6 : robot arm // 7 : end effector // 8,9 : gripper
        self._robot_joint_info = [p.getJointInfo(self._robot_body_id, i) for i in range(p.getNumJoints(self._robot_body_id))]
        self._robot_joint_indices = [x[0] for x in self._robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_lower_limit = [x[8] for x in self._robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._robot_joint_upper_limit = [x[9] for x in self._robot_joint_info if x[2] == p.JOINT_REVOLUTE]
        self._finger_joint_indices = [x[0] for x in self._robot_joint_info if x[2] == p.JOINT_PRISMATIC]
        self._finger_joint_lower_limit = [x[8] for x in self._robot_joint_info if x[2] == p.JOINT_PRISMATIC]
        self._finger_joint_upper_limit = [x[9] for x in self._robot_joint_info if x[2] == p.JOINT_PRISMATIC]
        
        
        # joint home config
        self._robot_home_joint_config = [0, math.pi*(-1/3), 0, math.pi*(-2/3), 0, math.pi*(5/6), math.pi*(1/4)]
        self._finger_home_joint_config = [0.04, 0.04] # [0.01, 0.01] # gripper : 0 ~ 0.04
        
        # Set maximum joint velocity. Maximum joint velocity taken from:
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=0, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=1, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=2, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=3, maxJointVelocity=150 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=4, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=5, maxJointVelocity=180 * (math.pi / 180))
        p.changeDynamics(bodyUniqueId=self._robot_body_id, linkIndex=6, maxJointVelocity=180 * (math.pi / 180))
        
        # Set coefficient of gripper
        p.changeDynamics(self._robot_body_id, 7,
                    lateralFriction=1,
                    spinningFriction=1,
                    rollingFriction=1,
                    frictionAnchor=True
                )
        p.changeDynamics(self._robot_body_id, 8,
                    lateralFriction=1,
                    spinningFriction=1,
                    rollingFriction=1,
                    frictionAnchor=True
                )
        p.changeDynamics(self._robot_body_id, 9,
                    lateralFriction=1,
                    spinningFriction=1,
                    rollingFriction=1,
                    frictionAnchor=True
                )
        self._joint_epsilon = 0.01
        self._robot_ee_idx = 7
        
        # self.num_demos = 1
        # add UI
        if self.enable_gui:
            self.eta_R_Id = p.addUserDebugParameter(" eta_R", 0.5, 5, 3)
            self.eta_p_Id = p.addUserDebugParameter(" eta_p", 0.5, 5, 3)
            self.diturbance_scale_Id = p.addUserDebugParameter(" disturbance_scale", 0, 1, 0)
            self.disturbance_time_Id = p.addUserDebugParameter(" disturbance_time", 0, 4, 0)
            self.start_Id = p.addUserDebugParameter(" start",1,0,0)
            self.restart_Id = p.addUserDebugParameter(" restart",1,0,0)
            self.restart_old = p.readUserDebugParameter(self.restart_Id)
        
        # model init
        self.model = None
        self.model_type = 'bc-deepovec'
        self.device = device
        
        # robot init
        self.reset()
        
        # offset
        self.offset4 = torch.tensor([0.007, 0, 0]).to(torch.float32)
    
    def load_demo(self, data_path, vis_demo=False):
        self.Ttraj_list = []
        self.Tdottraj_list = []
        self.dt_list = []
        
        data = torch.load(data_path, weights_only=False)
        
        for data_num in range(len(data)):
            Ttraj = torch.from_numpy(data[data_num]['Ttraj']).to(torch.float32)
            Tdottraj = torch.from_numpy(data[data_num]['Tdottraj']).to(torch.float32)
            dt = data[data_num]['dt']

            self.Ttraj_list.append(Ttraj)
            self.Tdottraj_list.append(Tdottraj)
            self.dt_list.append(dt)

            motion = self._motion.unsqueeze(0).repeat(len(Ttraj), 1, 1)
            demo_pos_np = (motion@Ttraj.to(torch.float32))[::10,:3,3].detach().numpy()
            
            if vis_demo:
                for i in range(len(demo_pos_np)-1):
                    p.addUserDebugLine(demo_pos_np[i], demo_pos_np[i+1], [0, 0, 0.3], 1, 0)
        
        self.Vb_max_list = []
        for d_num in range(len(self.Ttraj_list)):
            wbpdot_traj = se3.Tdot_to_wbpdot(self.Tdottraj_list[d_num], self.Ttraj_list[d_num])
            Vb_max = wbpdot_traj.norm(dim=-1).max() # same sacle Vb and wbpdot
            self.Vb_max_list.append(Vb_max)
        
        self.num_demos = len(data)
        if self.enable_gui:
            self.demo_num_Id = p.addUserDebugParameter(" demo_num", 0, self.num_demos-1, 0)

    
    def simulation(self, duration=None, vis_frame=False):
        if duration == None:
            duration = self._sampling_num
        for i in range(duration):
            if vis_frame:
                self.draw_frame(self._robot_ee_idx)
                self.draw_frame(self._robot_grasptarget_idx)
    
    def reset(self, speed=0.1):
        self.move_joints(self._robot_home_joint_config, speed)
        self.move_gripper(self._finger_home_joint_config, speed)
        self.simulation()
        ee_SE3 = self.get_ee_SE3()
        print(ee_SE3)
        
    def reload_env(self, position=[0, 0, 0]):
        p.resetSimulation()
        self._plane_id = p.loadURDF('assets/plane/plane.urdf', [0, 0, -0.055])
        if self.holder:
            self._umbrella_holder_id = p.loadURDF('assets/umbrella_holder/umbrella_holder.urdf',
                                                self._holder_init_position,
                                                self._holder_init_orientation,
                                                useFixedBase=True)
        self._robot_body_id = p.loadURDF("assets/franka_panda/panda_umbrella.urdf",
                                         position,
                                         p.getQuaternionFromEuler([0, 0, 0]),
                                         useFixedBase=True,
                                         flags=p.URDF_USE_SELF_COLLISION)
        self.reset()
        
    def move_joints(self, target_joint_state, blocking=False, speed=0.03):
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state, positionGains=speed * np.ones(len(self._robot_joint_indices)),
                                    forces=[500.]*len(self._robot_joint_indices))
        
        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon for i in
                           range(6)]):
                if time.time() - timeout_t0 > 5:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._robot_joint_indices]
                # time.sleep(0.001)
    
    def move_gripper(self, target_finger_state, blocking = False, speed=0.03):
        p.setJointMotorControlArray(self._robot_body_id, self._finger_joint_indices,
                                    p.POSITION_CONTROL, target_finger_state,
                                    positionGains=speed * np.ones(len(self._finger_joint_indices)))
        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_finger_state[i]) < self._joint_epsilon for i in range(2)]):
                if time.time() - timeout_t0 > 5:
                    p.setJointMotorControlArray(self._robot_body_id, self._finger_joint_indices, p.POSITION_CONTROL,[0.0, 0.0],
                                                positionGains=np.ones(len(self._finger_joint_indices)))
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, i)[0] for i in self._finger_joint_indices]
                # time.sleep(0.001)
    
    def grasp_object(self, blocking=False, force=2000, speed=0.03):
        # target joint state
        target_joint_state = np.array([0.0, 0.0])
        forces = np.array([force, force])

        # Move joints
        p.setJointMotorControlArray(
            self._robot_body_id, 
            self._finger_joint_indices,
            p.POSITION_CONTROL,
            target_joint_state,
            forces=forces,
            positionGains=speed * np.ones(len(self._finger_joint_indices))
        )

        # Block call until joints move to target configuration
        if blocking:
            timeout_t0 = time.time()
            while True:
                if time.time() - timeout_t0 > 1:
                    break
                # time.sleep(0.001)
    
    def move_link(self, position, orientation, link_idx, blocking=False, speed=0.03):
        # Use IK to compute target joint configuration
        target_joint_state = np.array(
            p.calculateInverseKinematics(self._robot_body_id, link_idx, position, orientation,
                                         maxNumIterations=10000, residualThreshold=.0001,
                                         lowerLimits=self._robot_joint_lower_limit,
                                         upperLimits=self._robot_joint_upper_limit))
        # Move joints
        p.setJointMotorControlArray(self._robot_body_id, self._robot_joint_indices, p.POSITION_CONTROL,
                                    target_joint_state[:-2], positionGains=speed * np.ones(len(self._robot_joint_indices)),
                                    forces=[500.]*len(self._robot_joint_indices))
        
        # Block call until joints move to target configuration
        if blocking:
            actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]

            timeout_t0 = time.time()
            while not all([np.abs(actual_joint_state[i] - target_joint_state[i]) < self._joint_epsilon 
                           for i in range(6)]):
                if time.time() - timeout_t0 > 5:
                    break
                actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
                
        actual_joint_state = [p.getJointState(self._robot_body_id, x)[0] for x in self._robot_joint_indices]
        return target_joint_state
    
    def move_link_SE3(self, target_SE3,  link_idx, blocking=False, speed=0.03):
        if not torch.is_tensor(target_SE3):
            target_SE3 = torch.tensor(target_SE3)
        if len(target_SE3.shape) == 2:
            target_SE3 = target_SE3.unsqueeze(0)
        target_quaternion_xyzw = lie.SO3_to_quaternion(target_SE3[:,:3,:3]).squeeze().numpy()
        target_position = target_SE3[:,:3,3].squeeze().numpy()
        target_joint_state = self.move_link(target_position, target_quaternion_xyzw, link_idx, blocking, speed)
        return target_joint_state
    
    def control_joint_vel(self, target_qdot):
        p.setJointMotorControlArray(bodyUniqueId=self._robot_body_id,
                            jointIndices=self._robot_joint_indices,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocities=target_qdot,
                            forces=[500] * len(target_qdot),)
        return None

    ############################################################################################################
    ############################################################################################################
    
    def move_ee(self, position, orientation, blocking):
        self.move_link(position, orientation, self._robot_ee_idx, blocking)
    
    def move_ee_SE3(self, target_SE3, blocking):
        self.move_link_SE3(target_SE3, self._robot_ee_idx, blocking)
    
    def move_grasptarget(self, position, orientation, blocking):
        self.move_link(position, orientation, self._robot_grasptarget_idx, blocking)
    
    def move_grasptarget_SE3(self, target_SE3, blocking):
        self.move_link_SE3(target_SE3, self._robot_grasptarget_idx, blocking)
    
    def disconnect(self):
        p.disconnect(self._physics_client_id)

    def gripper_open(self):
        self.move_gripper([0.04,0.04])
    
    def gripper_close(self):
        self.move_gripper([0.0, 0.0])
    
    def get_link_pose(self, link_index):
        link_state = p.getLinkState(self._robot_body_id, link_index) # (x,y,z,w)
        link_pos = link_state[0]
        link_ori = link_state[1] # (x,y,z,w)
        return link_pos, link_ori
    
    def get_link_SE3(self, link_index):
        link_pos, link_ori = self.get_link_pose(link_index)
        link_SE3 = np.eye(4)
        link_SE3[:3,3] = np.expand_dims(np.array(link_pos), axis=0)
        link_SE3[:3,:3] = lie.quaternion_to_SO3(torch.tensor(list(link_ori)).unsqueeze(0)).squeeze().numpy()
        return link_SE3
    
    def get_obj_pose(self, obj_id):
        obj_pos, obj_ori = p.getBasePositionAndOrientation(obj_id)
        return obj_pos, obj_ori
    
    def get_obj_SE3(self, obj_id):
        obj_pos, obj_ori = self.get_obj_pose(obj_id)
        obj_SE3 = np.eye(4)
        obj_SE3[:3,3] = np.expand_dims(np.array(obj_pos), axis=0)
        obj_SE3[:3,:3] = lie.quaternion_to_SO3(torch.tensor(list(obj_ori)).unsqueeze(0)).squeeze().numpy()
        return obj_SE3
    
    def get_ee_SE3(self):
        ee_SE3 = self.get_link_SE3(self._robot_ee_idx)
        return ee_SE3
    
    def get_grasptarget_SE3(self):
        grasptarget_SE3 = self.get_link_SE3(self._robot_grasptarget_idx)
        return grasptarget_SE3
    
    def get_joint_state(self):
        q = p.getJointStates(self._robot_body_id, self._robot_joint_indices)
        q = np.array([q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0]])
        return q
    
    def get_Js(self):
        S = ff.franka_screws_zero()
        q = p.getJointStates(self._robot_body_id, self._robot_joint_indices)
        q = np.array([q[0][0], q[1][0], q[2][0], q[3][0], q[4][0], q[5][0], q[6][0]])
        Js = ff.Jac_s(S, q)
        return Js
    
    def get_Jb(self):
        Js = self.get_Js()
        M = self.get_grasptarget_SE3()
        Jb = ff.Adjoint_T(ff.inverse_SE3(M)) @ Js
        return Jb
    
    def inv_vel_kinematics(self, Vb):
        Jb = self.get_Jb()
        try:
            qdot =  np.einsum('ij,j->i', lin.pinv(Jb), Vb)
        except:
            qdot = np.zeros(7)
        return qdot
    
    def forward_vel_kinematics(self, qdot):
        Jb = self.get_Jb()
        Vb = np.einsum('ij,j->i', Jb, qdot)
        return Vb
    
    def draw_frame(self, link_index, frame_length=0.1, life_time=1.):
        # Get the position and orientation of the link
        link_state = p.getLinkState(self._robot_body_id, link_index)
        link_pos = link_state[0]
        link_ori = link_state[1]

        # Create transformation matrix from link orientation
        rot_matrix = p.getMatrixFromQuaternion(link_ori)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Define frame arms
        x_axis = rot_matrix @ np.array([frame_length, 0, 0])
        y_axis = rot_matrix @ np.array([0, frame_length, 0])
        z_axis = rot_matrix @ np.array([0, 0, frame_length])

        # Draw the frame arms
        p.addUserDebugLine(link_pos, link_pos + x_axis, [1, 0, 0], lifeTime=life_time) # Red for X axis
        p.addUserDebugLine(link_pos, link_pos + y_axis, [0, 1, 0], lifeTime=life_time) # Green for Y axis
        p.addUserDebugLine(link_pos, link_pos + z_axis, [0, 0, 1], lifeTime=life_time) # Blue for Z axis 


    ############################################################################################################
    ########################################## for Vector field control ########################################
    
    def load_model(self, root, identifier, config_file, ckpt_file):
        self.model = load_pretrained(identifier, config_file, ckpt_file, root)[0].to(self.device)
        
    def model_forward(self, T_now, Ttraj, Tdottraj, eta_R, eta_p, demo_num):
        T_now_move = lie.inverse_SE3(self._motion.unsqueeze(0)) @ torch.tensor(T_now).unsqueeze(0)
        T_torch = T_now_move.to(self.device).type(torch.float32)
        Ttraj = torch.tensor(Ttraj).unsqueeze(0).to(self.device).to(torch.float32)
        model_output = self.model.forward(T_torch, Ttraj, eta_R, eta_p)
        Tdot = self._motion.unsqueeze(0).to(model_output) @ model_output
        Vb = lie.inverse_SE3(T_now.unsqueeze(0)) @ Tdot.to(T_now)
        Vb = lie.screw_bracket(Vb)[0].detach().cpu().numpy()
        
        Vb_max = self.Vb_max_list[demo_num] # norm is same for Vb and wbpdot
        Vbnorm_original = np.linalg.norm(Vb).clip(min=1e-7)
        Vbnorm_final = Vbnorm_original.clip(max=Vb_max*2).numpy()
        Vb = Vb / Vbnorm_original * Vbnorm_final
        
        return Vb
    
    def simulate_VF(self, vis_path=True, vis_frame=False):
        grasptarget_old = None
        
        while(True):
            start = p.readUserDebugParameter(self.start_Id)
            if start == 1:
                time_start = time.time()
                disturbed = False
                demo_num = round(p.readUserDebugParameter(self.demo_num_Id))
                T_init = self.Ttraj_list[demo_num][0].clone()
                if demo_num == 4:
                    T_init[:3,3] = T_init[:3,3] + self.offset4
                self.move_grasptarget_SE3(T_init, blocking=False)
                time.sleep(1)
                break
        while(True):
            # params
            eta_R = p.readUserDebugParameter(self.eta_R_Id)
            eta_p = p.readUserDebugParameter(self.eta_p_Id)
            demo_num = round(p.readUserDebugParameter(self.demo_num_Id))
            disturbance_scale = p.readUserDebugParameter(self.diturbance_scale_Id)
            disturbance_time = p.readUserDebugParameter(self.disturbance_time_Id)
            restart = p.readUserDebugParameter(self.restart_Id)
            
            # restart button
            if restart != self.restart_old:
                self.reload_env()
                T_init = Ttraj[0].clone()
                if demo_num == 4:
                    T_init[:3,3] = T_init[:3,3] + self.offset4
                self.move_grasptarget_SE3(T_init, blocking=False)
                time.sleep(1)
                time_start = time.time()
                disturbed = False
                self.restart_old = restart
            
            # apply disturbance
            time_now = time.time()
            if time_now - time_start > disturbance_time and not disturbed:
                if disturbance_scale > 0:
                    self.apply_SE3_disturbance(disturbance_scale)
                disturbed = True
            
            # set SE3
            Ttraj = self.Ttraj_list[demo_num].type(torch.float32)
            Tdottraj = self.Tdottraj_list[demo_num].type(torch.float32)
            T_now = self.get_grasptarget_SE3()
            
            T_now = torch.tensor(T_now).to(torch.float32).clone()
            
            # model forward
            if demo_num == 4:
                T_now[:3,3] = T_now[:3,3] - self.offset4
            Vb = self.model_forward(T_now, Ttraj, Tdottraj, eta_R, eta_p, demo_num)
            qdot = self.inv_vel_kinematics(Vb)
            self.control_joint_vel(qdot)
            
            # visualization
            if vis_frame:
                self.draw_frame(self._robot_grasptarget_idx)
            if vis_path:
                grasptarget_now = T_now[:3,3].numpy()
                if grasptarget_old is not None:
                    p.addUserDebugLine(grasptarget_old, grasptarget_now, [1, 0, 0], 1, 15)
                grasptarget_old = grasptarget_now
    
    def apply_SE3_disturbance(self, scale, sleep_time=0.5):
        c = 0.5
        d = 10
        if np.sqrt(scale**2 / (2*c)) < math.pi:
            R_scale = np.random.uniform(0, np.sqrt(scale**2 / (2*c)))
        else:
            R_scale = np.random.uniform(0, math.pi)
        
        R_dis = se3.SO3_uniform_sampling(batch_size=1)
        R_dis = torch.from_numpy(R_dis).to(torch.float32)
        w_dis = lie.log_SO3(R_dis.unsqueeze(0))
        w_dis_scaled = w_dis * (R_scale/np.linalg.norm(w_dis))
        R_dis_scaled = lie.exp_so3(w_dis_scaled)

        p_scale = torch.sqrt(torch.tensor((scale**2 - 2*c*R_scale**2) / d, dtype=torch.float32))
        p_dis = torch.randn(3, dtype=torch.float32)
        p_dis = p_dis * (p_scale / torch.norm(p_dis))
        
        T_dis = torch.eye(4)
        T_dis[:3,:3] = R_dis_scaled
        T_dis[:3,3] = p_dis
        
        T = self.get_grasptarget_SE3()
        T_target = T @ T_dis.detach().cpu().numpy()
        
        
        self.move_grasptarget_SE3(T_target, blocking=True)
        time.sleep(sleep_time)