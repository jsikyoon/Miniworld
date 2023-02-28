import numpy as np

class HeuristicPolicyHexagon:

    def __init__(self, dict):
        self.margin = dict['radius'] 
        '''
        st_matrix each column in left to right indicates
        turn_left, turn_right, move_forward, move_backward
        '''
        self.st_matrix = np.array([[0., 0., 0.5, -0.5], 
                                   [0., 0., 0., 0],
                                   [np.pi/3, -np.pi/3, 0, 0]])  
        

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
                # print(act, self.cur_pos)
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

