import numpy as np

class HeuristicPolicyRepeat:

    def __init__(self):
        self.goFlag = True

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
        assert ep_len % 2 == 0, "ep_len must be even"
        for _ in range(ep_len//2):
            agent_pos, agent_dir = obs['agent_pos'], obs['agent_dir']
            while True:
                act = np.random.randint(0, 4)
                obs, rew, _, _, _ = env.step(act)
                if np.sum(np.abs(obs['agent_pos'] - agent_pos)) + np.sum(np.abs(obs['agent_dir'] - agent_dir)) > 0: # if agent moves or rotates
                    break
            for k, v in obs.items():
                dict[k].append(v)
            dict['reward'].append(rew)
            dict['action'].append(act)
        actions_in_first_phase = dict['action'].copy()
        actions_in_first_phase.reverse()
        for step in range(ep_len//2):
            act_in_first_phase = actions_in_first_phase[step]
            if act_in_first_phase == 0:
                act = 1
            elif act_in_first_phase == 1:
                act = 0
            elif act_in_first_phase == 2:
                act = 3
            elif act_in_first_phase == 3:
                act = 2
            else:
                raise ValueError("act_in_first_phase must be 0, 1, 2, or 3")
            obs, rew, _, _, _ = env.step(act)
            for k, v in obs.items():
                dict[k].append(v)
            dict['reward'].append(rew)
            dict['action'].append(act)
    
        for k, v in dict.items():
            v_np = np.array(v)
            dict[k] = v_np
       
        return dict

    def act(self):
        
        action_seq = []  
        
        rotate_dir = np.random.randint(0, 2)  # 0: left, 1: right
        num_rotate = np.random.randint(1, 10)
        
        for _ in range(num_rotate):
            action_seq.append(rotate_dir)
            
        return np.array(action_seq)