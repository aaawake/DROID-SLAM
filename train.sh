# sudo rm -r /data/droidSLAM/datasets/SIAR/datasets/*
# echo "Transform rosbags to tum datasets"
# python rosbag2tum.py --bag_dir /data/droidSLAM/datasets/SIAR/rosbag

# echo "Transform rosbags to tum datasets"
# # python /home/dongjialin/workspace/pipeSLAM_ws/src/DROID-SLAM/tools/rosbag2tum.py --bag_files /data/droidSLAM/datasets/SIAR/rosbag/siar_2017-09-21-12-17-39.bag
# python /home/dongjialin/workspace/pipeSLAM_ws/src/DROID-SLAM/tools/rosbag2tum.py --bag_files /data/droidSLAM/datasets/SIAR/rosbag/siar_2017-10-11-11-05-03.bag
# python /home/dongjialin/workspace/pipeSLAM_ws/src/DROID-SLAM/tools/rosbag2tum.py --bag_files /data/droidSLAM/datasets/SIAR/rosbag/siar_2018-06-12-10-23-30.bag

echo "Train DROID-SLAM"
cd /home/dongjialin/workspace/pipeSLAM_ws/src/DROID-SLAM
python train.py --name batchSize_005 --ckpt checkpoints/droid.pth --datasets SIAR --datapath /data/droidSLAM/datasets/SIAR --gpus 1 --accumulation_step 32 | tee ./log/batchSize_005.log
--fmin 0.1 --fmax 8.0 --w1 1.0 --w2 10.0 --w3 5.0 

python train.py --name ckpt_002 --ckpt checkpoints/droid.pth --datapath /data/droidSLAM/datasets/abandonedfactory_sample_P001/ --gpus 1 --accumulation_step 1 | tee ./log/ckpt_002.log
