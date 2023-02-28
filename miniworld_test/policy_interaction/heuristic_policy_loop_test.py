import numpy as np

class HeuristicPolicyLoopTest:

    def __init__(self):
        pass

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
        
        # generate target pose for loop trajectory
        self.goal_poss = []
        x_dif = -np.sign(self.agent_pose_init[0])
        y_dif = -np.sign(self.agent_pose_init[1])
        
        for j in range(1, 5):
            goal1 = self.agent_pose_init + np.array([2*j*x_dif, 0, 0])
            goal2 = self.agent_pose_init + np.array([2*j*x_dif, 2*j*y_dif, 0])
            goal3 = self.agent_pose_init + np.array([0, 2*j*y_dif, 0])
            goal4 = self.agent_pose_init 
            self.goal_poss.append(goal1)
            self.goal_poss.append(goal2)
            self.goal_poss.append(goal3)
            self.goal_poss.append(goal4)
        self.goal_poss.append(goal1)

        # env interaction
        n = 0
        
        while not done:
            
            agent_pos, agent_dir = obs['agent_pos'][::2], 2*np.pi - obs['agent_dir']
            agent_angle = (agent_dir * 180 / np.pi) % 360
            agent_pose = np.concatenate([agent_pos, [agent_dir]])
            action_seq = self.act(agent_pose, n)
            n += 1

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

    def act(self, agent_pose, n):
        
        dist = np.linalg.norm(self.goal_poss[n][:2] - agent_pose[:2])
        agent_dir_wr_init = self.goal_poss[n][:2] - agent_pose[:2] 
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
        
        if angle_diff < 0:
            rot = 0
        elif angle_diff >= 0:
            rot =1
        
        num_rot = round(np.abs(angle_diff) / 15)
        action_rot_seq = np.array([rot] * num_rot)
        num_mov = round(dist / 0.5)     
        action_mov_seq = np.array([2] * num_mov)
        
        action_seq = np.concatenate([action_rot_seq, action_mov_seq])
        
        return action_seq   