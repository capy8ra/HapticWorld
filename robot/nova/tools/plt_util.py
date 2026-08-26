import matplotlib.pyplot as plt
import numpy as np

def plt_to_img(fig=None):
    if fig is None:
        fig = plt.gcf()
    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    img_array = np.asarray(rgba_buffer)
    return img_array
