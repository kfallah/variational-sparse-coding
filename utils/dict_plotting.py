"""
Plotting utility functions for learned sparse dictionary.

@Filename    dict_plotting
@Author      Kion 
@Created     5/29/20
"""
import matplotlib.animation as animation
import numpy as np
from matplotlib import pyplot as plt


def show_dict(phi, save_dir):
    """
    Create a figure for dictionary
    :param phi: Dictionary. Dimensions expected as pixels x num dictionaries
    """
    dict_mag = np.argsort(-1*np.linalg.norm(phi, axis=0))
    num_dictionaries = phi.shape[1]
    patch_size = int(np.sqrt(phi.shape[0]))
    fig = plt.figure(figsize=(12, 12))
    for i in range(num_dictionaries):
        plt.subplot(int(np.sqrt(num_dictionaries)), int(np.sqrt(num_dictionaries)), i + 1)
        dict_element = phi[:, dict_mag[i]].reshape(patch_size, patch_size)
        plt.imshow(dict_element, cmap='gray')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(save_dir, bbox_inches='tight')
    plt.close()

def show_phi_vid(phi_list):
    """
    Creates an HTML5 video for a list of dictionaries
    :param phi_list: List of dictionaries. Dimensions expected as time x pixels x num dictionaries
    :return: Matplotlib animation object containing HTML5 video for dictionaries over time
    """
    fig = plt.figure(figsize=(12, 12))
    num_dictionaries = phi_list.shape[2]
    patch_size = int(np.sqrt(phi_list.shape[1]))

    ax_list = []
    for p in range(num_dictionaries):
        ax_list.append(fig.add_subplot(int(np.sqrt(num_dictionaries)), int(np.sqrt(num_dictionaries)), p + 1))

    ims = []
    for i in range(phi_list.shape[0]):
        phi_im = []
        title = plt.text(0.5, .90, "Epoch Number {}".format(i),
                         size=plt.rcParams["axes.titlesize"],
                         ha="center", transform=fig.transFigure, fontsize=20)
        phi_im.append(title)
        for p in range(num_dictionaries):
            dict_element = phi_list[i, :, p].reshape(patch_size, patch_size)
            im = ax_list[p].imshow(dict_element, cmap='gray', animated=True)
            phi_im.append(im)
        ims.append(phi_im)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True, repeat_delay=2000)
    plt.close()
    return ani
