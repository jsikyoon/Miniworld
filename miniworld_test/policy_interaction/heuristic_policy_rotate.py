import numpy as np

class HeuristicPolicyRotate:

    def __init__(self):
        self.init_done = True
        self.flip = True

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
       
        #if self.flip:
        #    #act_seq = [0] * (ep_len//2) + [1] * (ep_len-ep_len//2)
        #    act_seq = [0] * (ep_len)
        #    self.flip = False
        #else:
        #    #act_seq = [1] * (ep_len//2) + [0] * (ep_len-ep_len//2)
        #    act_seq = [1] * (ep_len)
        #    self.flip = True
        #    
        #for act in act_seq:
        #    obs, rew, _, _, _ = env.step(act)
        #    for k, v in obs.items():
        #        dict[k].append(v)
        #    dict['reward'].append(rew)
        #    dict['action'].append(act)
        #    step += 1

        #    if step >= ep_len:
        #        done = True
        #        break
       
        # env interaction
        while not done:
            act_seq = self.act()
            for act in act_seq:
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

    # not used
    def act(self):

        action_seq = []

        rotate_dir = np.random.randint(0, 2)  # 0: left, 1: right
        num_rotate = np.random.randint(1, 10)

        for _ in range(num_rotate):
            action_seq.append(rotate_dir)

        return np.array(action_seq)
