import numpy as np
import copy

class HeuristicPolicyRandom:

    def __init__(self):
        pass

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
        
        # env interaction
        while not done:
            
            agent_pose_cur = self.get_pose(obs)
            action_seq = self.act(agent_pose_cur)
            
            for act in action_seq:
                
                pre_agent_pose = self.get_pose(obs)
                obs, rew, _, _, _ = env.step(act)
                nex_agent_pose = self.get_pose(obs)
                dist = np.linalg.norm(pre_agent_pose[:2] - nex_agent_pose[:2])
                angle_diff = ((pre_agent_pose[2] - nex_agent_pose[2]) * 180 / np.pi) % 360
                print('distance: ', dist, 'angle_diff: ', angle_diff)
                
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
        
        dist = np.linalg.norm(agent_pose[:2])
        if dist > 4:
            return np.array([0])
        else:
            action = np.random.randint(0, 3)
            return np.array([action])
  