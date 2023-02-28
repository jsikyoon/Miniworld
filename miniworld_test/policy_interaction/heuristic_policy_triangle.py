import numpy as np
import copy
import random

class HeuristicPolicyTriangle:

    def __init__(self, dict):
        self.num_mf = dict['num_mf']

    def get_pose(self, obs):
        # pose calculation. This makes x-y-theta in our common coordinate system.
        obs_copy = copy.deepcopy(obs)
        agent_pos, agent_dir = obs_copy['agent_pos'][::2], obs_copy['agent_dir'] 
        agent_pos[1] *= -1
        agent_pose = np.concatenate([agent_pos, [agent_dir]])
        return agent_pose
        

    def interaction(self, env, ep_len):
        
        # initialize
        dict = {}
        obs, info = env.reset()
        for k, v in obs.items():
            dict[k] = [v]
        dict['reward'] = [0.0]
        dict['action'] = [0]
        step = 0
        done = False
        
        # pose calculation. This makes x-y-theta in our common coordinate system.
        self.agent_pose_init = self.get_pose(obs)
        
        # get target poses with interaction
        targ_poses = {'0': self.agent_pose_init, '1': None, '2': None}
        action_seq0 = [2] * self.num_mf
        for act in action_seq0:
            obs, rew, _, _, _ = env.step(act)      
        targ_poses['1'] = self.get_pose(obs)
        action_seq1 = [0] * 8 + [2] * self.num_mf
        for act in action_seq1:
            obs, rew, _, _, _ = env.step(act)
        targ_poses['2'] = self.get_pose(obs)
        action_seq2 = [0] * 8 + [2] * self.num_mf + [0] * 8
        for act in action_seq2:
            obs, rew, _, _, _ = env.step(act)
        
        
        # env interaction  
        cur_idx = '0'
        
        while not done:
            
            obs_copy = copy.deepcopy(obs)
            agent_pos_cur, agent_dir_cur = obs_copy['agent_pos'][::2], obs_copy['agent_dir']
            agent_pos_cur[1] *= -1
            agent_angle_cur = (agent_dir_cur * 180 / np.pi) % 360
            agent_pose_cur = np.concatenate([agent_pos_cur, [agent_dir_cur]])

            '''
            Sample target pose
            '''
            cand = ['0', '1', '2']
            cand.remove(cur_idx)
            cur_idx = random.choice(cand)
            targ_pose = targ_poses[cur_idx]

            action_seq = self.act(agent_pose_cur, targ_pose)
            
            for act in action_seq:
                obs, rew, _, _, _ = env.step(act)
                
                for k, v in obs.items():
                    dict[k].append(v)
                dict['reward'].append(rew)
                dict['action'].append(act)
                step += 1
                
                if step >= ep_len:
                    done = True
                    break
                
        for k, v in dict.items():
            v_np = np.array(v)
            dict[k] = v_np

        return dict

    def act(self, agent_pose, targ_pose):
        '''
        Go to target pose
        '''
        dist = np.linalg.norm(targ_pose[:2] - agent_pose[:2])
        agent_dir_wr_init = targ_pose[:2] - agent_pose[:2] 
        agent_rad_wr_init = np.arctan2(agent_dir_wr_init[1], agent_dir_wr_init[0])  # -pi ~ +x(0) ~ +pi   
        agent_angle_wr_init = agent_rad_wr_init * 180 / np.pi % 360  # +x(0) ~ 360
        agent_angle = agent_pose[2] * 180 / np.pi % 360  # +x(0) ~ 360
        
        # coordinate system is x-y-theta (-<x>+, -vy^+, theta from +x-axis)
        # below code makes angle difference -180 ~ 180 w.r.t. agent angle
        if agent_angle_wr_init - agent_angle >= 180:
            angle_diff = agent_angle_wr_init - agent_angle - 360
        elif agent_angle_wr_init - agent_angle < 180 and agent_angle_wr_init - agent_angle >= 0:
            angle_diff = agent_angle_wr_init - agent_angle
        elif agent_angle_wr_init - agent_angle < 0 and agent_angle_wr_init - agent_angle >= -180:
            angle_diff = agent_angle_wr_init - agent_angle
        else: #  agent_angle_wr_init - agent_angle < -180
            angle_diff = agent_angle_wr_init - agent_angle + 360

        if angle_diff >= 90:
            rot = 0
            mov = 2
        elif angle_diff < 90 and angle_diff >= 0:
            rot = 0
            mov = 2
        elif angle_diff < 0 and angle_diff >= -90:
            rot = 1
            mov = 2
        else:  # angle_diff < -90
            rot = 1
            mov = 2
        
        num_rot = round(np.abs(angle_diff) / 15)
        action_rot_seq = np.array([rot] * num_rot)
        num_mov = round(dist / 0.5)     
        action_mov_seq = np.array([mov] * num_mov)
        
        action_seq = np.concatenate([action_rot_seq, action_mov_seq])
        return np.array(action_seq)