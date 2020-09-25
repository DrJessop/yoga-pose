import os

if os.getcwd().split('/')[-1] != 'yoga-pose':
    raise Exception('Ensure that you are running this script from the yoga-pose root directory')

os.system('git clone https://github.com/facebookresearch/VideoPose3D')
os.mkdir('./VideoPose3D/checkpoint')
os.mkdir('./VideoPose3D/npz')
os.system('python -m pip install git+https://github.com/facebookresearch/detectron2.git')
os.mkdir('videos')
os.mkdir('videos/input')
os.mkdir('videos/output')
