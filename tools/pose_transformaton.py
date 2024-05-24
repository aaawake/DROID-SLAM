import os
from tqdm import tqdm
import numpy as np
import tf.transformations as tf

class poseTrans():

    def __init__(self, pose_dir, new_pose_dir, tf_dir):
        # camera_t = [0.0, -0.045, 0.0]
        # camera_r = [0.0, 0.0, 0.0, 1.0]
        # camera_matrix = tf.quaternion_matrix(camera_r)
        # camera_matrix[:3, 3] = camera_t

        # odom_t = [69.31824232114981, -160.28337562593032, 1.524931932776855]
        # odom_r = [0.011122724663730824, -0.023129196525927848, 0.9908491336636917, 0.13251158282215014]
        # odom_matrix = tf.quaternion_matrix(odom_r)
        # odom_matrix[:3, 3] = odom_t

        # base_t = [-31.899148723820307, 14.602922608454081, 0.0]
        # base_r = [0.047388294995123946, 0.015469786779874121, -0.2008510691621877, 0.9783526374527982]
        # base_matrix = tf.quaternion_matrix(base_r)
        # base_matrix[:3, 3] = base_t

        # ele_t = [0.0004999999999999999, 0.0, 0.35]
        # ele_r = [0.0, 0.0, 0.0, 1.0]
        # ele_matrix = tf.quaternion_matrix(ele_r)
        # ele_matrix[:3, 3] = ele_t

        # baselink_matrix = np.dot(odom_matrix, base_matrix, ele_matrix)

        # self.baselink_matrix = np.array([[-7.84108790e-01, -6.19847264e-01,  3.10286244e-02,  9.62470322e+01], \
        #                                  [ 6.20588693e-01, -7.82555665e-01,  4.97624882e-02, -1.82718079e+02], \
        #                                  [-6.56351638e-03,  5.82752179e-02,  9.98278979e-01, -5.55111512e-16], \
        #                                  [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        # self.camera_matrix = np.array([[ 1.   ,  0.   ,  0.   ,  0.   ], \
        #                                [ 0.   ,  1.   ,  0.   , -0.045], \
        #                                [ 0.   ,  0.   ,  1.   ,  0.   ], \
        #                                [ 0.   ,  0.   ,  0.   ,  1.   ]])
        self.pose_dir = pose_dir
        self.new_pose_dir = new_pose_dir
        self.tf_dir = tf_dir


    def read_tf(self):
        frame_transforms = {}
        with open(self.tf_dir, 'r') as file:
            lines = file.readlines()

        for line in lines:
            data = line.split()
            frame_id = data[0]
            child_frame_id = data[1]
            translation = [float(data[i]) for i in range(2, 5)]
            rotation = [float(data[i]) for i in range(5, 9)]
            frame_transforms[(frame_id, child_frame_id)] = {
                'translation': translation,
                'rotation': rotation
            }
        return frame_transforms

    def base_transform(self, frame_transforms, frame_id):
        odom_matrix = tf.quaternion_matrix(frame_transforms[('map', 'odom')]['rotation'])
        odom_matrix[:3, 3] = frame_transforms[('map', 'odom')]['translation']

        base_matrix = tf.quaternion_matrix(frame_transforms[('/odom', '/base_link')]['rotation'])
        base_matrix[:3, 3] = frame_transforms[('/odom', '/base_link')]['translation']

        ele_matrix = tf.quaternion_matrix(frame_transforms[('/base_link', 'electronics_center')]['rotation'])
        ele_matrix[:3, 3] = frame_transforms[('/base_link', 'electronics_center')]['translation']

        camera_matrix = tf.quaternion_matrix(frame_transforms[('/'+frame_id, '/'+frame_id[:-4]+'rgb_frame')]['rotation'])
        camera_matrix[:3, 3] = frame_transforms[('/'+frame_id, '/'+frame_id[:-4]+'rgb_frame')]['translation']

        baselink_matrix = np.dot(np.dot(ele_matrix, base_matrix), odom_matrix)

        return ele_matrix, camera_matrix

    def trans_pose(self):
        # Creating a conversion matrix
        frame_transforms = self.read_tf()
        frame_id = "electronics_center"
        child_frame_id = self.new_pose_dir.split('/')[-2] + '_link'
        tf_translation = frame_transforms[(frame_id, child_frame_id)]['translation']
        tf_rotation = frame_transforms[(frame_id, child_frame_id)]['rotation']
        tf_matrix = tf.quaternion_matrix(tf_rotation)
        tf_matrix[:3, 3] = tf_translation

        ele_matrix, camera_matrix = self.base_transform(frame_transforms, child_frame_id)

        print(f"Transforming to {child_frame_id}")
        with open(self.pose_dir, 'r') as file:
            lines = file.readlines()

        with open(self.new_pose_dir, 'w') as new_file:
            # for i in tqdm(range(len(lines)), bar_format='Transform: {percentage:.1f}%|{bar}|', leave=True):
                # line = lines[i]
            for line in lines:
                data = line.split()
                timestamp = data[0]
                pose1_translation = [float(data[i]) for i in range(1, 4)]
                pose1_rotation = [float(data[i]) for i in range(4, 8)]
                pose1_matrix = tf.quaternion_matrix(pose1_rotation)
                pose1_matrix[:3, 3] = pose1_translation

                # pose2_matrix = np.dot(camera_matrix, np.dot(tf_matrix, np.dot(baselink_matrix, pose1_matrix)))
                # pose2_matrix = np.dot(camera_matrix, np.dot(tf_matrix, np.dot(ele_matrix, pose1_matrix)))
                pose2_matrix = np.dot(tf_matrix, np.dot(ele_matrix, pose1_matrix))
                # pose2_matrix = np.dot(ele_matrix, pose1_matrix)
                pose2_translation = pose2_matrix[:3, 3]
                pose2_rotation_quaternion = tf.quaternion_from_matrix(pose2_matrix)
                translation_str = " ".join([f"{value:.7f}" for value in pose2_translation])
                rotation_str = " ".join([f"{value:.7f}" for value in pose2_rotation_quaternion])

                new_file.write(f"{timestamp} {translation_str} {rotation_str}\n")