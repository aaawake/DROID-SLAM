
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
import tf.transformations as tf

cur_path = osp.dirname(osp.abspath(__file__))
test_split = osp.join(cur_path, 'tartan_test.txt')
test_split = open(test_split).read().split()


class SIAR(RGBDDataset):

    # scale depths to balance rot & trans
    DEPTH_SCALE = 1.0

    def __init__(self, mode='training', **kwargs):
        self.mode = mode
        self.n_frames = 2
        super(SIAR, self).__init__(name='SIAR_'+str(SIAR.DEPTH_SCALE), **kwargs)
        # super(SIAR, self).__init__(name='SIAR', crop_size=[96, 128], **kwargs)

    @staticmethod 
    def is_test_scene(scene):
        # print(scene, any(x in scene for x in test_split))
        return any(x in scene for x in test_split)

    def pose_transform(self, poses):
        # 右前上->右下前
        # T_b_c = np.array([[1,  0,  0, 0],
        #                   [0,  0, -1, 0],
        #                   [0,  1,  0, 0],
        #                   [0,  0,  0, 1]])
        # T_c_b = np.array([[ 1,  0, 0, 0],
        #                   [ 0,  0, 1, 0],
        #                   [ 0, -1, 0, 0],
        #                   [ 0,  0, 0, 1]])

        # 前左上->右下前
        T_c_b = np.array([[0, -1,  0, 0],
                          [0,  0, -1, 0],
                          [1,  0,  0, 0],
                          [0,  0,  0, 1]])
        T_b_c = np.array([[ 0,  0, 1, 0],
                          [-1,  0, 0, 0],
                          [ 0, -1, 0, 0],
                          [ 0,  0, 0, 1]])
        camera_poses = np.ones_like(poses[:, 1:])

        T_w_c = np.eye(4)
        for i in range(len(poses)):
            quaternion = poses[i, 4:]
            rotation = tf.quaternion_matrix(quaternion)
            translation = poses[i, 1:4]            
            T_w_b = np.eye(4)
            T_w_b[:3, :3] = rotation[:3, :3]
            T_w_b[:3, 3] = translation.T

            if i > 0:
                T_bn_bo = np.dot(np.linalg.inv(T_w_b), T_w_b_last)
                T_cn_co = np.dot(np.dot(T_c_b, T_bn_bo), T_b_c)
                T_w_c = np.dot(T_w_c_last, np.linalg.inv(T_cn_co))

            T_w_b_last = T_w_b
            T_w_c_last = T_w_c

            camera_poses[i][:3] = T_w_c[:3, 3].T
            camera_poses[i][3:] = tf.quaternion_from_matrix(T_w_c).T

        return np.around(camera_poses, 5)

    def _build_dataset(self):
        from tqdm import tqdm
        print("Building SIAR dataset with DEPTH_SCALE is " + str(SIAR.DEPTH_SCALE))
        logging.info("Building SIAR dataset with DEPTH_SCALE is " + str(SIAR.DEPTH_SCALE))

        scene_info = {}
        # scenes = glob.glob(osp.join(self.root, '*/*/*/*'))
        scenes = glob.glob(osp.join(self.root, 'SIAR/datasets/*'))
        # for i in tqdm(len(sorted(scenes)), bar_format='Reading dataset: {percentage:.1f}%|{bar}|', leave=True):
        #     scene = scenes[i]
        for scene in tqdm(sorted(scenes)):
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
                poses = self.pose_transform(np.round(poses))
                # np.savetxt('/home/dongjialin/data/droidSLAM/datasets/SIAR/back/newPose20.txt', poses, fmt='%.7f %.7f %.7f %.7f %.7f %.7f %.7f', delimiter='\n')
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
        t, tx, ty, tz, rx, ry, rz, rw = np.split(poses, 8, axis=1)
        poses = np.concatenate((-ty, -tz, tx, -ry, -rz, rx, rw), axis=1)

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
        t, tx, ty, tz, rx, ry, rz, rw = np.split(poses, 8, axis=1)
        poses = np.concatenate((-ty, -tz, tx, -ry, -rz, rx, rw), axis=1)

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