import os
import argparse

# Run this in home directory of VideoPose3D
parser = argparse.ArgumentParser()
parser.add_argument('input_file')
parser.add_argument('output_file')
args = parser.parse_args()
input_file = args.input_file
output_file = args.output_file

# Keypoint detection
os.chdir('inference')
os.system('python infer_video_d2.py \
    --cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml \
    --output-dir ../output_directory \
    --image-ext mp4 \
    ../videos')

# Dataset preparation
os.chdir('../data')
os.system('python prepare_data_2d_custom.py -i ../output_directory -o myvideos')

# 3D reconstruction via back-projection
os.chdir('..')
os.system('python run.py -d custom -k myvideos -arc 3,3,3,3,3 -c checkpoint --evaluate pretrained_h36m_cpn.bin --render --viz-subject {} --viz-action custom --viz-camera 0 --viz-video videos/{} --viz-output {} --viz-size 6'.format(input_file, input_file, output_file))
