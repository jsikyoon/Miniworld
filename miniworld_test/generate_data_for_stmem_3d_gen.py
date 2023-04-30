'''
This is sample code for generating data.
The environment is just single room with x,y in [-5,5,-5,5].
We can control the spawn position of the agent and procedurally generated wall textures.

policy type
    - rotate: Agent spawns in [-3,3,-3,3]. Agent only rotates for random steps (~U(1,10)) in random direction.
    - mfmb: Agent spawns in [-3,3,-3,3]. Agent only moves forward and backward (randomly) not hitting the wall.

# command
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m miniworld_test.generate_data
'''
import sys
import pyglet
pyglet.options['headless'] = True
import gymnasium as gym
import miniworld
import numpy as np
from pathlib import Path
from miniworld_test.policy_interaction import *
import skvideo.io
from einops import rearrange
import matplotlib.pyplot as plt
from PIL import Image
import imageio

def generate_video(memory_image, gt_image, query_image, dir, idx):

    imageio.mimsave(f'{dir}/memory_{idx}.gif', memory_image, fps=4)
    for i in range(len(gt_image)):
        imageio.mimsave(f'{dir}/gt{i}_{idx}.gif', gt_image[i], fps=4)
    for i in range(len(query_image)):
        Image.fromarray(query_image[i]).save(f'{dir}/query{i}_{idx}.png')

# setting
total_num_ep = 10000
st_num_ep = sys.argv[1]
nd_num_ep = sys.argv[2]

data_dir = f'./datasets/room_with_without_objects_stmem_3d_gen_various_angle3'
Path(data_dir).mkdir(parents=True, exist_ok=True)
data_type = {'train': int(total_num_ep * 0.9),
             'eval': int(total_num_ep * 0.1)}

# generate data
data_type = {'train': int(total_num_ep * 0.9), 'eval': int(total_num_ep * 0.1)}
env_with_objects = gym.make(f"MiniWorld-RoomObjectsSTMEM-v0")
# observations with objects
for type, num_ep in data_type.items():
    for j in range(num_ep):
        if type == 'eval':
            seed = data_type['train'] + j
        else:
            seed = j
        if seed < int(st_num_ep) or seed > int(nd_num_ep):
            continue
        dict = {}
        # observations with objects
        obss_with_objects, dirs_with_objects = [], []
        obs, info = env_with_objects.reset(seed=seed)
        obss_with_objects.append(obs['image'])
        dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(18): # 225 -> 135
            obs, _, _, _, _ = env_with_objects.step(1) # turn right
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(18): # 135 -> 45
            obs, _, _, _, _ = env_with_objects.step(1) # turn right
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(9): # 45 -> 0
            obs, _, _, _, _ = env_with_objects.step(1) # turn right
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(54): # 18 x 3
            obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(9): # 0 -> 315
            obs, _, _, _, _ = env_with_objects.step(1) # turn right
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(18): # 315 -> 225
            obs, _, _, _, _ = env_with_objects.step(1) # turn right
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        dict['memory_image'] = np.array(obss_with_objects)[:-1] # (n, 84, 84, 3)
        dict['memory_dir'] = np.array(dirs_with_objects)[:-1]

        # get ground truth images
        obss_with_objects, dirs_with_objects = [], []
        def collect_gt_images(obs):
            gt_images, gt_dirs = [], []
            while True:
                if env_with_objects.action_start():
                    break
                obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            gt_images.append(obs['image'])
            gt_dirs.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
            for _ in range(5): # collect 5 images
                obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
                gt_images.append(obs['image'])
                gt_dirs.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
            return gt_images, gt_dirs
        # front 180
        env_with_objects.set_agent_dir(dir=180*np.pi/180)
        obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
        gt_images, gt_dirs = collect_gt_images(obs)
        obss_with_objects.append(gt_images)
        dirs_with_objects.append(gt_dirs)
       
        # right 90
        env_with_objects.set_agent_dir(dir=90*np.pi/180)
        obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
        gt_images, gt_dirs = collect_gt_images(obs)
        obss_with_objects.append(gt_images)
        dirs_with_objects.append(gt_dirs)
        
        # back 0 (multiple images)
        env_with_objects.set_agent_dir(dir=0*np.pi/180)
        for i in range(4):
            env_with_objects.replace_back_object(i)
            gt_images, gt_dirs = collect_gt_images(obs)
            obss_with_objects.append(gt_images)
            dirs_with_objects.append(gt_dirs)
        
        # left 270
        env_with_objects.set_agent_dir(dir=270*np.pi/180)
        gt_images, gt_dirs = collect_gt_images(obs)
        obss_with_objects.append(gt_images)
        dirs_with_objects.append(gt_dirs)
        dict["gt_image"] = np.array(obss_with_objects)
        dict["gt_dir"] = np.array(dirs_with_objects)
        
        # back to origin position
        env_with_objects.set_agent_dir(dir=225*np.pi/180)

        # observations without objects
        def get_initial_state():
            while True:
                if env_with_objects.action_start():
                    break
                obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            return obs
        
        #env_with_objects.remove_objects()
        obss_without_objects, dirs_without_objects = [], []
        # front
        env_with_objects.set_agent_dir(dir=180*np.pi/180)
        obs = get_initial_state()
        obss_without_objects.append(obs['image'])
        dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
       
        # right
        env_with_objects.set_agent_dir(dir=90*np.pi/180)
        obs = get_initial_state()
        obss_without_objects.append(obs['image'])
        dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        
        # back (back is with object at the beginning)
        env_with_objects.set_agent_dir(dir=0*np.pi/180)
        #env_with_objects.rollback_objects()
        for i in range(4):
            env_with_objects.replace_back_object(i)
            while True:
                if env_with_objects.action_start():
                    break
                obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            obs, _, _, _, _ = env_with_objects.step(2) # move forward (not working)
            obss_without_objects.append(obs['image'])
            dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        #env_with_objects.remove_objects()
        
        # left
        env_with_objects.set_agent_dir(dir=270*np.pi/180)
        obs = get_initial_state()
        obss_without_objects.append(obs['image'])
        dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
 
        dict['query_image'] = np.array(obss_without_objects)
        dict['query_dir'] = np.array(dirs_without_objects)

        Path(f'{data_dir}/{type}').mkdir(parents=True, exist_ok=True)
        # save first 10 videos
        if type == 'train' and j < 10:
            generate_video(
                dict['memory_image'].copy().astype(np.uint8),
                dict['gt_image'].copy().astype(np.uint8),
                dict['query_image'].copy().astype(np.uint8),
                data_dir, j)
        print(f'saving {type}/{j}-th episode in npz...')
        np.savez(f'{data_dir}/{type}/{j}.npz', **dict)
