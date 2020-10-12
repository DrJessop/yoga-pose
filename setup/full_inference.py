import os
import subprocess
import sys
import argparse

from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='''Input file, ( should be in ./videos/input/ ). ex. If your 
                                             input file is vid.mp4, the path ./videos/input/vid.mp4 should
                                             exist.''')
parser.add_argument('-o', '--output', help='Output file name, will be saved to ./videos/output/<output_file>')
parser.add_argument('-j', '--joints', help='Name of numpy array where 3d joints will be stored')

if len(sys.argv) == 1:  # If no arguments supplied, defaults to help message
    parser.print_help(sys.stderr)
    sys.exit(1)

args        = parser.parse_args()
input_file  = args.input
output_file = args.output
joints      = args.joints
ext         = input_file.split('.')[-1]  # extension of input_file 

if os.getcwd().split('/')[-1] != 'yoga-pose':
    raise Exception('Ensure that you are running this script from the yoga-pose root directory')

# Go into VideoPose3D home directory
if 'VideoPose3D' not in os.listdir():
    raise Exception('''Ensure that you have run the videopose_setup.py script to download VideoPose3D 
                       and to setup detectron2.''')

os.chdir('VideoPose3D')

if 'pretrained_h36m_detectron_coco.bin' not in os.listdir('checkpoint'):
    raise Exception('You must first install the checkpoint model! See {} for download link.'.format(
        'https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin'
       )
    )

# Keypoint detection
os.chdir('inference')
logger.info('Beginning keypoint detection in directory {}'.format(os.getcwd()))
try:
    subprocess.run(['python3', 'infer_video_d2.py', '--cfg', 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', 
                    '--output-dir', '../npz', '--image-ext', ext, '../../videos/input/{}'.format(input_file)], check=True)
except subprocess.CalledProcessError as e:
    logger.info(e.output)
    sys.exit(1)

# Dataset preparation
os.chdir('../data')
logger.info('Beginning 2D dataset preparation')
try:
    subprocess.run(['python3', 'prepare_data_2d_custom.py', '-i', '../npz', '-o', 'myvideos'], check=True)
except subprocess.CalledProcessError as e:
    logger.info(e.output)
    sys.exit(1)

# 3D reconstruction via back-projection
os.chdir('..')
logger.info('Beginning 3D reconstruction')

try:
    subprocess.run(['python3', 'run.py', '-d', 'custom', '-k', 'myvideos', '-arc', '3,3,3,3,3', '-c', 'checkpoint',
                            '--evaluate', 'pretrained_h36m_detectron_coco.bin',
                            '--render', '--viz-subject', input_file, '--viz-action', 'custom',
                            '--viz-camera', '0', '--viz-video', '../videos/input/{}'.format(input_file), 
                            '--viz-output', '../videos/output/{}'.format(output_file),
                            '--viz-size', '6', '--viz-export', '../joints/{}'.format(joints)], check=True)
except subprocess.CalledProcessError as e:
    logger.info(e)
    sys.exit(1)
