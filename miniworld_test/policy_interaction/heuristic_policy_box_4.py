import numpy as np
import copy
import random

class HeuristicPolicyBox4:

    def __init__(self, dict):
        self.box = dict['box']

    # pose calculation. This makes x-y-theta in our common coordinate system.
    def get_pose(self, obs):
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
        
        self.agent_pose_init = self.get_pose(obs)
        self.targ_poses = {'0': self.agent_pose_init, '1': None, '2': None, '3': None}
        low_list = [[self.agent_pose_init[0]-self.box, self.agent_pose_init[1], 0.],
                    [self.agent_pose_init[0], self.agent_pose_init[1], 0.],
                    [self.agent_pose_init[0]-self.box, self.agent_pose_init[1]-self.box, 0.]]
        high_list = [[self.agent_pose_init[0], self.agent_pose_init[1]+self.box, 0.],
                     [self.agent_pose_init[0]+self.box, self.agent_pose_init[1]+self.box, 0.],
                     [self.agent_pose_init[0]+self.box, self.agent_pose_init[1], 0.]]
                                            
        for i in range(3):
            self.targ_poses[str(i+1)] = self.sample_pose(low_list[i], high_list[i])
        
        # env interaction
        cur_idx = '0'
        
        while not done:
            
            agent_pose_cur = self.get_pose(obs)
            cand = ['0', '1', '2', '3']
            cand.remove(cur_idx)
            cur_idx = random.choice(cand)
            targ_pose = self.targ_poses[cur_idx]
        
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
        Go to random pose
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

        '''
        if angle_diff >= 90:
            angle_diff -= 180
            rot = 0
            mov = 3
        elif angle_diff < 90 and angle_diff >= 0:
            rot = 1
            mov = 2
        elif angle_diff < 0 and angle_diff >= -90:
            rot = 0
            mov = 2
        else:  # angle_diff < -90
            angle_diff += 180
            rot = 1
            mov = 3
        '''
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
    
    def sample_pose(self, low, high):
        
        while True:
            rand_pose = np.random.uniform(low=low, high=high) 
            dist = np.linalg.norm(rand_pose[:2] - self.agent_pose_init[:2])
            if dist > 1.5:
                # print(rand_pose - self.agent_pose_init)
                break

        return rand_pose