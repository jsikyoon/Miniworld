import numpy as np

class HeuristicPolicyMFMB:

    def __init__(self):
        self.margin = 1.5

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
        
        # env interaction
        
        while not done:
            
            agent_pos, agent_dir = obs['agent_pos'], obs['agent_dir']
            agent_angle = (agent_dir * 180 / np.pi) % 360
            action_seq = self.act(agent_pos, agent_angle)
            
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

    def act(self, agent_pos, agent_angle):
        
        action_seq = []  
        
        dist = np.linalg.norm(agent_pos - self.agent_pos_init)
        
        if dist < self.margin:
            
            # random move
            move_dir = np.random.randint(2, 4)  # 2: move forward, 3: move backward
            num_move = np.random.randint(1, 4)
            for _ in range(num_move):
                action_seq.append(move_dir)    
            self.anti_dir = 3 if move_dir == 2 else 2
        
        else:
            
            # move to spawn position
            for _ in range(4):
                action_seq.append(self.anti_dir)
            
        return np.array(action_seq)

