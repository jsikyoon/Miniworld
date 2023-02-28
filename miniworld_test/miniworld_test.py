#!/usr/bin/env python3
'''
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m miniworld_test.miniworld_test
'''
import time
import gymnasium as gym
import miniworld
import numpy as np
import matplotlib.pyplot as plt
import skvideo.io


# Benchmark loading time
st = time.time()
env = gym.make("MiniWorld-CustomV0-v0")
# env.seed(0)
obs, info = env.reset()

step = 0

poss = [obs['agent_pos']]
imgs = [obs['image']]
top_down_imgs = [obs['top_camera']]

while True:
    '''
    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |
    | 3   | move back                   |
    | 4   | pick up                     |
    | 5   | drop                        |
    | 6   | toggle / activate an object |
    | 7   | complete task               |
    '''
    
    
    action = 0
    obs, reward, termination, truncation, info = env.step(action)
    
    # print(obs.shape, np.max(obs), np.min(obs))
    # print(reward)
    # print(termination)
    # print(truncation)
    for k, v in obs.items():
        print(k, v.shape)

    poss.append(obs['agent_pos'])
    imgs.append(obs['image'])
    top_down_imgs.append(obs['top_camera'])
    print(action, obs['agent_pos'], obs['agent_dir'])
    
    if termination or truncation:
        env.reset()
    
    if step > 300:
        break
    step += 1


imgs = np.array(imgs)
top_down_imgs = np.array(top_down_imgs)
video = np.concatenate((imgs, top_down_imgs), axis=2)

fps = 4
crf = 17
vid_out = skvideo.io.FFmpegWriter('./video.mp4', 
            inputdict={'-r': str(fps)},
            outputdict={'-r': str(fps), '-c:v': 'libx264', '-crf': str(crf), 
                        '-preset': 'ultrafast', '-pix_fmt': 'yuv444p'}
)

for frame in video:
    vid_out.writeFrame(frame)
vid_out.close()

env.close()
