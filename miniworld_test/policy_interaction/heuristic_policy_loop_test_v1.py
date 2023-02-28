import numpy as np
import copy

class HeuristicPolicyLoopTestV1:

    def __init__(self):
        self.nr = 5
        self.nc = 5
        self.gap = 2
        self.tar_poss = np.random.permutation([False] * (self.nr-1) + [True] * (self.nc-1))
        self.ret_poss = np.random.permutation([False] * (self.nr-1) + [True] * (self.nc-1))

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
        obs_copy = copy.deepcopy(obs)
        agent_pos_init, agent_dir_init = obs_copy['agent_pos'][::2], obs_copy['agent_dir'] 
        agent_pos_init[1] *= -1
        agent_angle_init = (agent_dir_init * 180 / np.pi) % 360
        self.agent_pose_init = np.concatenate([agent_pos_init, [agent_dir_init]])
        
        # env interaction
        index = 0
        targ_pose = self.agent_pose_init.copy()
        
        while not done:
            
            obs_copy = copy.deepcopy(obs)
            agent_pos_cur, agent_dir_cur = obs_copy['agent_pos'][::2], obs_copy['agent_dir']
            agent_pos_cur[1] *= -1
            agent_angle_cur = (agent_dir_cur * 180 / np.pi) % 360
            agent_pose_cur = np.concatenate([agent_pos_cur, [agent_dir_cur]])

            '''
            Sample target pose
            '''
            if index < self.nr + self.nc - 2:  # go to bottom-right
                if self.tar_poss[index]:  # go down
                    targ_pose -= np.array([0, 12, 0])
                else:  # go right
                    targ_pose += np.array([12, 0, 0])
            elif index < 2*(self.nr + self.nc - 2):  # back to top-left (initial pose)
                if self.ret_poss[index - (self.nr + self.nc - 2)]:  # go up
                    targ_pose += np.array([0, 12, 0])
                else:  # go left
                    targ_pose -= np.array([12, 0, 0])
            else:
                targ_pose = None
            index += 1

            action_seq = self.act(agent_pose_cur, targ_pose)
            
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

    def act(self, agent_pose, targ_pose):
        if targ_pose is None:
            return np.array([0])
        else:
            '''
            Go to target pose
            '''
            dist = np.linalg.norm(targ_pose[:2] - agent_pose[:2])
            agent_dir_wr_init = targ_pose[:2] - agent_pose[:2] 
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