from kornia import depth
import numpy as np 
import matplotlib.pyplot as plt
import os
import imageio



class VideoRecorder(object):
    def __init__(self, save_dir, file_name):
        self.n = 0
        self.save_dir = save_dir
        self.pc_film = []
        self.pix_film = []
        self.depth_film = []
        self.file_name = file_name

    def reset(self):
        self.pc_film = []
        self.pix_film = []
        self.depth_film = []

    def calback(self, pc_frame, pix_frame, depth_frame):
        self.pc_film.append(pc_frame)
        self.pix_film.append(pix_frame)
        self.depth_film.append(depth_frame)

    def save(self):
        path_pc = os.path.join(self.save_dir + '/pc', 'pc_video' + f'{self.n}')
        path_pix = os.path.join(self.save_dir + '/pix', 'pix_video' + f'{self.n}')
        path_depth = os.path.join(self.save_dir + '/depth', 'depth_video' + f'{self.n}')

        imageio.mimsave(path_pc, self.pc_film, fps=25)
        imageio.mimsave(path_pix, self.pc_film, fps=25)

        self.n += 1

