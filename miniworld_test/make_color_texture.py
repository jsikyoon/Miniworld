import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as cl

wall_tex_name_list = ['brick_wall', 'grass', 'lava', 'rock',
                      'stucco', 'water', 'wood', 'wood_planks']

for name in wall_tex_name_list:
    img_dir = f'../miniworld/textures/{name}_{1}.png'
    img = plt.imread(img_dir)
    print(name, img.shape, np.max(img), np.min(img))

wall_tex_name_list = ['blue', 'orange', 'green', 'red', 'purple',
                      'brown', 'pink', 'olive']

for name in wall_tex_name_list:
    color = cl.to_rgba_array(name)
    print(name, color[0])
    image = np.zeros([512, 512, 4])
    image[..., :] = color[0]
    plt.imsave(f'../miniworld/textures/{name}_{1}.png', image)


# map = cl.get_named_colors_mapping()
# print(map)
