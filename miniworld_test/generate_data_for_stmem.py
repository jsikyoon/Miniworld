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

def generate_video(image, dir, idx):

    total_steps, h, w, c = image.shape  # _, 84(h=y), 84(w=x)

    # make video
    video = image
    fps = '4'
    crf = '17'
    vid_out = skvideo.io.FFmpegWriter(f'{dir}/video_{idx}.mp4',
                inputdict={'-r': fps},
                outputdict={'-r': fps, '-c:v': 'libx264', '-crf': crf,
                            '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
    )

    for frame in video:
        vid_out.writeFrame(frame)
    vid_out.close()


# setting
total_num_ep = 5000
data_dir = f'./datasets/room_with_without_objects_stmem'
Path(data_dir).mkdir(parents=True, exist_ok=True)
data_type = {'train': int(total_num_ep * 0.9),
             'eval': int(total_num_ep * 0.1)}

# generate data
data_type = {'train': int(total_num_ep * 0.9), 'eval': int(total_num_ep * 0.1)}
env_with_objects = gym.make(f"MiniWorld-RoomObjectsSTMEM-v0")
env_without_objects = gym.make(f"MiniWorld-RoomSTMEM-v0")
# observations with objects
for type, num_ep in data_type.items():
    if type == 'eval':
        seed = data_type['train'] + num_ep
    else:
        seed = num_ep
    for j in range(num_ep):
        dict = {}
        # observations with objects
        obss_with_objects, dirs_with_objects = [], []
        obs, info = env_with_objects.reset(seed=seed)
        obss_with_objects.append(obs['image'])
        dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(23): # 360 / 15
            obs, _, _, _, _ = env_with_objects.step(0) # turn left
            obss_with_objects.append(obs['image'])
            dirs_with_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        # observations without objects
        obss_without_objects, dirs_without_objects = [], []
        obs, info = env_without_objects.reset(seed=seed)
        obss_without_objects.append(obs['image'])
        dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))
        for _ in range(23): # 360 / 15
            obs, _, _, _, _ = env_without_objects.step(0) # turn left
            obss_without_objects.append(obs['image'])
            dirs_without_objects.append(round(obs['agent_dir']/(2*np.pi)*360 % 360))

        dict['image'] = np.array(obss_with_objects + obss_without_objects)
        dict['agent_dir'] = np.array(dirs_with_objects + dirs_without_objects)
        Path(f'{data_dir}/{type}').mkdir(parents=True, exist_ok=True)
        # save first 10 videos
        if type == 'train' and j < 10:
            generate_video(dict['image'].copy().astype(np.uint8), data_dir, j)
            exit(1)
        print(f'saving {type}/{j}-th episode in npz...')
        np.savez(f'{data_dir}/{type}/{j}.npz', **dict)
