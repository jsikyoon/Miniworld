import numpy as np

class HeuristicPolicyLocalNoWall:

    def __init__(self, dict):
        self.margin = dict['radius']
        '''
        st_matrix each column in left to right indicates
        turn_left, turn_right, move_forward, move_backward
        '''
        self.st_matrix = np.array([[0., 0., 0.5, -0.5], 
                                   [0., 0., 0., 0],
                                   [np.pi/12, -np.pi/12, 0, 0]]) 

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
        self.agent_pos_init = obs['agent_pos']
        self.cur_pos = np.array([0.0, 0.0, 0.0])
        
        # env interaction
        
        while not done:
            
            agent_pos, agent_dir = obs['agent_pos'], obs['agent_dir']
            agent_angle = (agent_dir * 180 / np.pi) % 360
            action_seq = self.act()
            
            for act in action_seq:
                obs, rew, _, _, _ = env.step(act)
                # print(act, obs['agent_pos'], (obs['agent_dir'] * 180 / np.pi) % 360)
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

    def act(self):
        
        action_seq = []    
        
        # check whether next_pos is out of margin
        while True:
            rand_action = np.random.randint(4)
            rand_action_1hot = np.zeros([4,])
            rand_action_1hot[rand_action] = 1.0
            rot_matrix = np.array([[np.cos(self.cur_pos[-1]), -np.sin(self.cur_pos[-1]), 0],
                                   [np.sin(self.cur_pos[-1]), np.cos(self.cur_pos[-1]), 0],
                                   [0, 0, 1]])
            
            next_pos = np.matmul(rot_matrix, np.matmul(self.st_matrix, rand_action_1hot)) + self.cur_pos
            dist = np.linalg.norm(next_pos[:2])
            if dist < self.margin:
                action_seq.append(rand_action)
                break
        
        # update self.cur_pos
        rot_matrix = np.array([[np.cos(self.cur_pos[-1]), -np.sin(self.cur_pos[-1]), 0],
                               [np.sin(self.cur_pos[-1]), np.cos(self.cur_pos[-1]), 0],
                               [0, 0, 1]])
        self.cur_pos = np.matmul(rot_matrix, np.matmul(self.st_matrix, rand_action_1hot)) + self.cur_pos
        
        return np.array(action_seq)
    


        '''
        dist = np.linalg.norm(agent_pos - self.agent_pos_init)
        
        if dist < self.margin:
            
            # random action
            rand_move = np.random.randint(4)
            action_seq.append(rand_move)
            # action_seq.append(rand_move)
        
        else:
            
            # move to center
            agent_dir_wr_init = agent_pos - self.agent_pos_init     
            agent_rad_wr_init = np.arctan2(-agent_dir_wr_init[2], agent_dir_wr_init[0])  # -pi ~ +x(0) ~ +pi   
            agent_angle_wr_init = agent_rad_wr_init * 180 / np.pi % 360  # +x(0) ~ 360
            desired_angle = 180 + agent_angle_wr_init if agent_angle_wr_init < 180 else agent_angle_wr_init - 180
            # print(agent_dir_wr_init, agent_angle_wr_init)
            
            if agent_angle_wr_init < 180:  # 1,2 quadrant
                if agent_angle > agent_angle_wr_init and agent_angle < agent_angle_wr_init + 180:
                    action = 0
                    angle_diff = desired_angle - agent_angle
                    action_ = 1  # anti-action 
                    angle_diff_ = 360 - angle_diff  # anti-angle
                else:
                    action = 1
                    angle_diff = agent_angle - desired_angle if desired_angle < agent_angle else 360 - (desired_angle - agent_angle)
                    action_ = 0  # anti-action 
                    angle_diff_ = 360 - angle_diff  # anti-angle
            else:  # 3, 4 quadrant
                if agent_angle > agent_angle_wr_init - 180 and agent_angle < agent_angle_wr_init:
                    action = 1
                    angle_diff = agent_angle - desired_angle
                    action_ = 0  # anti-action 
                    angle_diff_ = 360 - angle_diff  # anti-angle
                else:
                    action = 0
                    angle_diff = desired_angle - agent_angle if desired_angle > agent_angle else 360 - (agent_angle - desired_angle)
                    action_ = 1  # anti-action 
                    angle_diff_ = 360 - angle_diff  # anti-angle
    
            
            # num_rot_max = (angle_diff) // 15 + 1
            # num_rot = np.random.randint(np.max([0, num_rot_min]), num_rot_max)
            
            rand_rot_dir = np.random.randint(2)
            if rand_rot_dir:
                num_rot = int(angle_diff / 15)
                num_move = int(dist / 0.5) + 1    
                for i in range(num_rot):
                    action_seq.append(action) 
                for i in range(num_move):
                    action_seq.append(2)
            else:
                num_rot = int(angle_diff_ / 15)
                num_move = int(dist / 0.5) + 1
                for i in range(num_rot):
                    action_seq.append(action_) 
                for i in range(num_move):
                    action_seq.append(2)
            
            # print(desired_angle, angle_diff, num_rot)
        '''