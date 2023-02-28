import numpy as np

class HeuristicPolicyStick:

    def __init__(self, dict):
        self.height = dict['height']
        self.width = dict['width']

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
        
        agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
        agent_angle = (agent_dir * 180 / np.pi) % 360
        agent_pose = np.concatenate([agent_pos, [agent_dir]])
        self.agent_pose_init = agent_pose
        self.goFlag = True
        
        # env interaction
        
        while not done:
            
            agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
            agent_angle = (agent_dir * 180 / np.pi) % 360
            agent_pose = np.concatenate([agent_pos, [agent_dir]])
            # print('start_pos', agent_pose[:2], agent_angle)
            action_seq = self.act(agent_pose)
            
            for act in action_seq:
                obs, rew, _, _, _ = env.step(act)
                agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
                agent_angle = (agent_dir * 180 / np.pi) % 360
                agent_pose = np.concatenate([agent_pos, [agent_dir]])
                # print(agent_pose[:2], agent_angle)
                
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

    def act(self, agent_pose):

        '''
        Sample random pose
        '''
        if self.goFlag:
            self.goFlag = False
            rand_pose_or = np.random.uniform(low=[self.height, -self.width, 0.],
                                             high=[2*self.height, self.width, 0.],)    
        else:
            self.goFlag = True
            rand_pose_or = np.random.uniform(low=[-self.height, -self.width, 0.],
                                             high=[-2*self.height, self.width, 0.],)
        
        rot_matrix = np.array([[np.cos(self.agent_pose_init[-1]), -np.sin(self.agent_pose_init[-1]), 0],
                               [np.sin(self.agent_pose_init[-1]), np.cos(self.agent_pose_init[-1]), 0],
                               [0, 0, 1]])
        rand_pose = np.matmul(rot_matrix, rand_pose_or) + self.agent_pose_init
        
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

        # print('agent_pose', agent_pose[:2], (agent_pose[2] * 180 / np.pi) % 360)
        # print('goal_pose', rand_pose[:2], (rand_pose[2] * 180 / np.pi) % 360)
        # print(angle_diff, dist)

        # num_rot = int(np.abs(angle_diff) / 15)
        num_rot = np.random.randint(1, 5)
        action_rot_seq = np.random.permutation([0] * num_rot + [1] * num_rot)
        
        num_mov = int(dist / 0.5)     
        action_mov_seq = np.array([mov] * num_mov)
        
        action_seq = np.concatenate([action_mov_seq, action_rot_seq])
        return action_seq