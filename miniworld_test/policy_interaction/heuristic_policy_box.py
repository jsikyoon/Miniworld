import numpy as np
import copy

class HeuristicPolicyBox:

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
        
        # env interaction
        
        while not done:
            
            agent_pose_cur = self.get_pose(obs)
            # print('cur_pos: ', agent_pose_cur)
            action_seq = self.act(agent_pose_cur)
            
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

        # save data
        for k, v in dict.items():
            # image
            # top_camera
            # agent_pos
            # agent_dir
            # reward
            # action
            if k == 'image' or k == 'top_camera':
                v_np = np.array(v).astype(np.uint8)
            else:
                v_np = np.array(v).astype(np.float32)
            dict[k] = v_np
        return dict

    def act(self, agent_pose):

        '''
        Sample random pose
        '''
        rand_pose = np.random.uniform(low=[self.agent_pose_init[0]-self.box,
                                          self.agent_pose_init[1]-self.box, 0.],
                                     high=[self.agent_pose_init[0]+self.box,
                                           self.agent_pose_init[1]+self.box, 0.],)
        # print('tar_pose: ', rand_pose)
        '''
        Go to random pose
        '''
        dist = np.linalg.norm(rand_pose[:2] - agent_pose[:2])
        agent_dir_wr_init = rand_pose[:2] - agent_pose[:2] 
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
            rot = 1
            mov = 3
        elif angle_diff < 90 and angle_diff >= 0:
            rot = 0
            mov = 2
        elif angle_diff < 0 and angle_diff >= -90:
            rot = 1
            mov = 2
        else:  # angle_diff < -90
            angle_diff += 180
            rot = 0
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