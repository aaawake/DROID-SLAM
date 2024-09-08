import rosbag
import cv2
import numpy as np
import os
import glob
from shutil import copyfile
from geometry_msgs.msg import PoseStamped
from tqdm import tqdm
from pose_transformaton import poseTrans
from align_timestamp import alignTimestamp

def rosbag2tum(args):
    for i in tqdm(range(len(args.bag_files)), bar_format='Processing rosbags: {percentage:.1f}%|{bar}|', leave=True):
        bag_file = args.bag_files[i]
    # bag_files = glob.glob(os.path.join(args.bag_dir, "*"))
    # for bag_file in bag_files:
        print(f'Processing {os.path.basename(bag_file)}')
        store_bag_dir = os.path.join(args.store_dir, os.path.basename(bag_file)[:-4])
        if not os.path.exists(store_bag_dir):
            os.mkdir(store_bag_dir)

        # bag = rosbag.Bag(bag_file, 'r')
        # frame_id_pairs = []
        # for topic, msg, t in tqdm(bag.read_messages(), desc="messages"):
        #     if topic.split('/')[-1] == 'compressedDepth' or topic.split('/')[-1] == 'compressed':
        #         camera_type_ = topic.strip('/').split('/')[0]
        #         if camera_type_ == "back_left" or camera_type_ == "back_right" or camera_type_ == "front_left" or camera_type_ == "front_right" or camera_type_ == "up":
        #             continue

        #         timestamp = msg.header.stamp.to_sec()
        #         # set the path for storing files
        #         camera_dir = os.path.join(store_bag_dir, camera_type_)
        #         camera_type = topic.strip('/').split('/')[1]
        #         pic_filename = os.path.join(camera_dir, f"{camera_type}/{timestamp}")
        #         if not os.path.exists(os.path.join(camera_dir, camera_type)):
        #             os.makedirs(os.path.join(camera_dir, camera_type))
        #         # if not os.path.exists(os.path.join(camera_dir, 'depth')):
        #         #     os.makedirs(os.path.join(camera_dir, 'depth'))
        #         # store picture data
        #         if camera_type == 'rgb':
        #             img_data = np.frombuffer(msg.data, np.uint8)
        #             img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        #             # cv2.imwrite(pic_filename+'.png', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #             cv2.imwrite(pic_filename+'.png', img)
        #             with open(os.path.join(camera_dir, f'{camera_type}.txt'), 'a') as rgb_file:
        #                 rgb_file.write(f"{timestamp} {pic_filename}.png\n")
        #         elif camera_type == 'depth_registered':
        #             raw_data = msg.data
        #             # Remove the header from the raw data
        #             depth_header_size = 12
        #             raw_data = raw_data[depth_header_size:]
        #             depth_img_raw = cv2.imdecode(np.frombuffer(raw_data, np.uint8), cv2.IMREAD_UNCHANGED)
        #             # cv2.imwrite(os.path.join(camera_dir, f"depth/{timestamp}")+'.png', depth_img_raw)
        #             depth_data = np.array(depth_img_raw)
        #             np.save(pic_filename, depth_data)
        #             with open(os.path.join(camera_dir, f'{camera_type}.txt'), 'a') as rgb_file:
        #                 rgb_file.write(f"{timestamp} {pic_filename}.npy\n")
        #     elif topic == '/baseline':
        #         timestamp = msg.header.stamp.to_sec()
        #         pose = msg.pose
        #         tx = pose.position.x
        #         ty = pose.position.y
        #         tz = pose.position.z
        #         qx = pose.orientation.x
        #         qy = pose.orientation.y
        #         qz = pose.orientation.z
        #         qw = pose.orientation.w
        #         with open(os.path.join(store_bag_dir, 'pose.txt'), 'a') as gd_file:
        #             gd_file.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")
        #     elif topic == '/tf':
        #         frame_id = msg.transforms[0].header.frame_id
        #         child_frame_id = msg.transforms[0].child_frame_id
        #         if (frame_id, child_frame_id) not in frame_id_pairs:
        #             frame_id_pairs.append((frame_id, child_frame_id))
        #             t = msg.transforms[0].transform.translation
        #             r = msg.transforms[0].transform.rotation
        #             with open(os.path.join(store_bag_dir, 'tf.txt'), 'a') as tf_file:
        #                 tf_file.write(f"{frame_id} {child_frame_id} {t.x} {t.y} {t.z} {r.x} {r.y} {r.z} {r.w}\n")
        #     else:
        #         continue
        # bag.close()

        # Transforms the pose to each camera frame
        print(f"Compute all camera poses of {os.path.basename(bag_file)} and align timestamp")
        dirs_of_all_camera_pos = [directory for directory in glob.glob(os.path.join(store_bag_dir, '*')) if os.path.isdir(directory)]
        old_pose_dir = os.path.join(store_bag_dir, 'pose.txt')
        for directory in dirs_of_all_camera_pos:
            if os.path.basename(directory) != 'front':
                pt = poseTrans(pose_dir=old_pose_dir, \
                            new_pose_dir=os.path.join(directory, 'pose.txt'), \
                            tf_dir=os.path.join(store_bag_dir, 'tf.txt'))
                pt.trans_pose()
            # else:
            #     copyfile(old_pose_dir, os.path.join(directory, 'pose.txt'))
                # continue
            alignT = alignTimestamp(first=os.path.join(directory, 'pose.txt'), \
                                    second=os.path.join(directory, 'rgb.txt'), \
                                    third=os.path.join(directory, 'depth_registered.txt'), \
                                    max_difference=0.04)
            alignT.align()
        print("\n")

    print("\nAll data has been read!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bag_files', nargs='+', help="path to rosbags")
    parser.add_argument('--bag_dir', help="path to rosbags")
    parser.add_argument('--store_dir', default='/data/droidSLAM/datasets/SIAR/datasets', help="Path to storage files")
    args = parser.parse_args()
    rosbag2tum(args)