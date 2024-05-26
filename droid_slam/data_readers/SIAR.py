
import numpy as np
import torch
import glob
import cv2
import os
import os.path as osp
import logging

from lietorch import SE3
from .base import RGBDDataset
from .stream import RGBDStream

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class SIAR(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 5.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(SIAR, self).__init__(name='SIAR', **kwargs)
        # super(SIAR, self).__init__(name='SIAR', crop_size=[96, 128], **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building SIAR dataset")
        logging.info("Building SIAR dataset")

        scene_info = {}
        # scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        scenes = glob.glob(osp.join(self.root, 'datasets/*'))
        # for i in tqdm(len(sorted(scenes)), bar_format='Reading dataset: {percentage:.1f}%|{bar}|', leave=True):
        #     scene = scenes[i]
        for scene in scenes:
            cameras = glob.glob(osp.join(scene, "*"))
            for camera in cameras:
                if not os.path.isdir(camera):
                    continue
                images = sorted(glob.glob(osp.join(camera, 'rgb/*.png')))
                depths = sorted(glob.glob(osp.join(camera, 'depth_registered/*.npy')))
                
                poses = np.loadtxt(osp.join(camera, 'pose.txt'), delimiter=' ')
                # X = forward / backward; Y = left / right; Z = up / down
                # to
                # X = right / left; Y = down / up; Z = forward / backward
                poses = poses[:, [2, 3, 1, 5, 6, 4, 7]]
                # poses = poses[:, [-2, -3, 1, 4, 5, 6, 7]]
                poses[:,:3] /= SIAR.DEPTH_SCALE
                intrinsics = [SIAR.calib_read(osp.basename(camera))] * len(images)

                # graph of co-visible frames based on flow
                graph = self.build_frame_graph(poses, depths, intrinsics)

                camera = '/'.join(camera.split('/'))
                scene_info[camera] = {'images': images, 'depths': depths, 
                    'poses': poses, 'intrinsics': intrinsics, 'graph': graph}

        return scene_info

    @staticmethod
    def calib_read(camera):
        # save every camera intrinsics as a dict
        intrinsics_depth = {
            "back": [570.3, 570.3, 314.5, 235.5],
            "back_left": [285.2, 285.2, 157.0, 117.5], 
            "back_right": [285.2, 285.2, 157.0, 117.5], 
            "front": [570.3, 570.3, 314.5, 235.5], 
            "front_left": [285.2, 285.2, 157.0, 117.5], 
            "front_right": [285.2, 285.2, 157.0, 117.5], 
            "up": [570.3, 570.3, 314.5, 235.5],
        }
        intrinsics_rgb = {
            "back": [570.3, 570.3, 319.5, 239.5], 
            "back_left": [285.2, 285.2, 159.5, 119.5], 
            "back_right": [285.2, 285.2, 159.5, 119.5], 
            "front": [570.3, 570.3, 319.5, 239.5], 
            "front_left": [285.2, 285.2, 159.5, 119.5], 
            "front_right": [285.2, 285.2, 159.5, 119.5], 
            "up": [285.2, 285.2, 159.5, 119.5], 
        }
        return np.array(intrinsics_rgb[camera])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

    @staticmethod
    def depth_read(depth_file):
        depth = np.load(depth_file) / SIAR.DEPTH_SCALE
        depth[depth==np.nan] = 1.0
        depth[depth==np.inf] = 1.0
        return depth

# For inferencing and evaluation, no change yet
class SIARStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(SIARStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/SIAR'

        scene = osp.join(self.root, self.datapath)
        image_glob = osp.join(scene, 'image_left/*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(scene, 'pose_left.txt'), delimiter=' ')
        # X = forward / backward; Y = right / left; Z = up / down
        # to
        # X = right / left; Y = up / down; Z = forward / backward
        # poses = poses[:, [2, 3, 1, 5, 6, 4, 7]]
        poses = poses[:, [-2, -3, 1, 4, 5, 6, 7]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)

# For inferencing and evaluation, no change yet
class SIARTestStream(RGBDStream):
    def __init__(self, datapath, **kwargs):
        super(SIARTestStream, self).__init__(datapath=datapath, **kwargs)

    def _build_dataset_index(self):
        """ build list of images, poses, depths, and intrinsics """
        self.root = 'datasets/mono'
        image_glob = osp.join(self.root, self.datapath, '*.png')
        images = sorted(glob.glob(image_glob))

        poses = np.loadtxt(osp.join(self.root, 'mono_gt', self.datapath + '.txt'), delimiter=' ')
        # poses = poses[:, [2, 3, 1, 5, 6, 4, 7]]
        poses = poses[:, [-2, -3, 1, 4, 5, 6, 7]]

        poses = SE3(torch.as_tensor(poses))
        poses = poses[[0]].inv() * poses
        poses = poses.data.cpu().numpy()

        intrinsic = self.calib_read(self.datapath)
        intrinsics = np.tile(intrinsic[None], (len(images), 1))

        self.images = images[::int(self.frame_rate)]
        self.poses = poses[::int(self.frame_rate)]
        self.intrinsics = intrinsics[::int(self.frame_rate)]

    @staticmethod
    def calib_read(datapath):
        return np.array([320.0, 320.0, 320.0, 240.0])

    @staticmethod
    def image_read(image_file):
        return cv2.imread(image_file)