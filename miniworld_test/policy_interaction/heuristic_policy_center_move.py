import numpy as np

class HeuristicPolicyCenterMove:

    def __init__(self, dict):
        self.margin = dict['radius']

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
        agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
        self.agent_angle_init = (agent_dir * 180 / np.pi) % 360
        agent_pose = np.concatenate([agent_pos, [agent_dir]])
        self.agent_pose_init = agent_pose
        
        # env interaction
        
        while not done:
            
            agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
            agent_angle = (agent_dir * 180 / np.pi) % 360
            agent_pose = np.concatenate([agent_pos, [agent_dir]])
            action_seq = self.act(agent_pose)

            for act in action_seq:
                obs, rew, _, _, _ = env.step(act)
                # agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
                # agent_angle = (agent_dir * 180 / np.pi) % 360
                # agent_pose = np.concatenate([agent_pos, [agent_dir]])
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
        
        dist = np.linalg.norm(self.agent_pose_init[:2] - agent_pose[:2])
        
        if dist < self.margin:
            
            action_seq = np.array([np.random.randint(4)])  # random policy
        
        else:
            
            agent_dir_wr_init = self.agent_pose_init[:2] - agent_pose[:2] 
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

            # print('agent_angle', agent_angle)
            # print('agent_angle_wr_init', agent_angle_wr_init)
            # print('angle_diff', angle_diff)
            
            rot = np.random.randint(2)
            if rot == 1:
                if angle_diff < 0:
                    angle_diff += 360
            else:
                if angle_diff >= 0:
                    angle_diff -= 360
            
            num_rot = round(np.abs(angle_diff) / 15)
            action_rot_seq = np.array([rot] * num_rot)
            num_mov = round(dist / 0.5)     
            action_mov_seq = np.array([2] * num_mov)
            
            action_seq = np.concatenate([action_rot_seq, action_mov_seq])
        
        return action_seq   