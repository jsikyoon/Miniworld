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

def generate_video(image, top_camera, agent_pos, obs_steps, dir, idx):

    total_steps, h, w, c = image.shape  # _, 84(h=y), 84(w=x)

    # make video
    video = np.concatenate((image, top_camera), axis=2)
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

    # save traj scenes
    n_r = int(np.sqrt(total_steps))
    n_c = total_steps // n_r
    n_vis = n_r * n_c
    image_tosave = rearrange(image[:n_vis],
                             '(n_r n_c) h w c -> (n_r h) (n_c w) c',
                             n_r=n_r, n_c=n_c)
    plt.imsave(f'{dir}/image_{idx}.png', image_tosave)


# setting

env_type = 'V0'
env = gym.make(f"MiniWorld-Custom{env_type}-v0")
policy_type = 'rotate'
type_mapper = {
    'random': [HeuristicPolicyRandom, 50, 100, 100, None],
    # for a basic dataset
    'mfmb': [HeuristicPolicyMFMB, 70, 100, 5000, None],
    'rotate': [HeuristicPolicyRotate, 50, 80, 5000, None],
    #'rotate': [HeuristicPolicyRotate, 23, 30, 5000, None],
    'repeat': [HeuristicPolicyRepeat, 70, 100, 20000, None],
}

add_input = type_mapper[policy_type][4]
if add_input is not None:
    actor = type_mapper[policy_type][0](add_input)
else:
    actor = type_mapper[policy_type][0]()
obs_steps = type_mapper[policy_type][1]
total_steps = type_mapper[policy_type][2]
total_num_ep = type_mapper[policy_type][3]
assert total_steps > obs_steps
data_dir = f'./datasets/offline_miniworld_{env_type}_{policy_type}'
Path(data_dir).mkdir(parents=True, exist_ok=True)
data_type = {'train': int(total_num_ep * 0.9),
             'eval': int(total_num_ep * 0.1)}
num_action = env.action_space.n


# generate data
for type, num_ep in data_type.items():

    Path(f'{data_dir}/{type}').mkdir(parents=True, exist_ok=True)

    for j in range(num_ep):

        # actor interact with env for total_steps

        dict = actor.interaction(env, total_steps)

        # save first 3 videos

        if type == 'train' and j < 10:

            generate_video(dict['image'].copy().astype(np.uint8),
                           dict['top_camera'].copy().astype(np.uint8),
                           dict['agent_pos'].copy(),
                           obs_steps, data_dir, j)

        print(f'saving {type}/{j}-th episode in npz...')
        # save only first image of top_camera
        dict['top_camera'] = dict['top_camera'][0:1]
        np.savez(f'{data_dir}/{type}/{j}_{total_steps}.npz', **dict)
