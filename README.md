# Yoga Pose

During the pandemic, a lot of fitness-related activities went fully online, including yoga. I'm sure many fellow yogis have felt the frustration of not getting feedback on their posture during this time, which was the motivation for creating this piece of software. In this tutorial, you will be building an application that helps correct yoga poses using React, Flask, and Facebook's VideoPose3D. 

In order, this tutorial will cover 

<ol>
  <li>Building the backend for the yoga app, which will include...
    <ol>
      <li>Creating a Flask app</li>
      <li>Using REDIS for job queues</li>
      <li>Using Facebook's VideoPose3D library for performing 3D pose estimation on an instructor and student video</li>
      <li>Establishing an error function between the student and instructor poses</li>
    </ol>
  </li>
  <li>Building the React frontend, which will include...
    <ol>
      <li>Overview of the structure of a React application</li>
      <li>Using React routers for a single-page application</li>
      <li>How to upload videos and submit requests to the flask backend</li>
      <li>How to display processed videos to the screen</li>
    </ol>
  </li>
</ol>

<p align="center">
  <img width="400" height="400" src="https://github.com/DrJessop/yoga-pose/blob/staging/app/images/vinyasa.gif?raw=true">
</p>

This is a beginner tutorial for <b>React</b> and <b>Pytorch</b>, however there are some pre-requisites. Prior to this tutorial, you should be comfortable with:

<ul>
  <li>Python3+</li>
  <li>Flask (beginner knowledge)</li>
  <li>Javascript, HTML, and CSS (optional)</li>
</ul>

## Getting Started
Before beginning, you need to install the following:
<ul>
  <li>node package manager (npm)</li>
  <li>python3+</li>
</ul>

### Installing npm, python3+, and Redis on Mac
NOTE: Ensure that you have homebrew installed!

```sh
brew install node
brew install python

wget http://download.redis.io/redis-stable.tar.gz
tar xvzf redis-stable.tar.gz
cd redis-stable
make
```

### Required python modules

The requirements file for all necessary Python modules can be found in the root folder as requirements.txt.

To install, in terminal, run

```sh
python3 -m pip install -r requirements.txt
```

### Installing VideoPose3D and Detectron2
To set up VideoPose3D and Detectron2, run the following commands from the project root:

```sh
git clone https://github.com/facebookresearch/VideoPose3D
mkdir ./VideoPose3D/checkpoint
mkdir ./VideoPose3D/npz
python -m pip install git+https://github.com/facebookresearch/detectron2.git
mkdir videos
mkdir videos/input
mkdir videos/output
```

NOTE: Installing Detectron2 requires you to have gcc and g++ installed. 

Once this is successfully completed, a particular model needs to be downloaded from https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin. Once downloaded, place the file in ./VideoPose3D/checkpoint.

### Running model on a CPU (Optional)
If you don't have access to a GPU, you must change infer_video_d2.py in VideoPose3D/inference/:

Make sure that you import torch, ie...

```python
...
import torch
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
...
```

And in the main function of infer_video_d2.py, change the following...

```python
...
def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)
        
    predictor = DefaultPredictor(cfg)
    ...
```

to...

```python
def main(args):

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)

    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cpu'
        
    predictor = DefaultPredictor(cfg)
    ...
```

This will ensure that if this code is run on a machine without a GPU, the model will run inference on a CPU.

### Creating React app
The next step is to create our React application. In the frontend folder, copy the package.json and package-lock.json files from https://github.com/DrJessop/yoga-pose/tree/staging/app/frontend, and then run

```sh
npm install
```

This will install all the node modules required for this project.

### Directory structure
Below is a diagram of the directory structure of the entire project. Using this structure, create empty files with the following names, as the rest of the tutorial will be referring to these files.

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/directory_structure.png?raw=true)

## Building python backend 
### Overview
In this section, you will be building a Flask API backed By VideoPose3D which will provide the necessary functions to correct a student's pose based off of their instructor. 

Pose estimation refers to estimating joint key-points on a subject and connecting them together. In 3D pose estimation, the depth of the joints is also estimated (how far from the camera each joint is). Here is an example of 3D video pose estimation for a yoga clip:

The process that this app will use for pose estimation is simple:

<ol>
  <li>Given a video of an instructor and student, extract the joint coordinates for each of the student and instructor across all frames of the respective videos</li>
  <li>Compute the angles between all pairs of adjacent limbs (ex. the angle between your right forearm and your right upper arm) for both the student and the instructor</li>
  <li>Compute the absolute differences across all of these limb pairs, as well as the sum of angles across each frame</li>
  <li>Find the best possible match of student and instructor using point set rigid registration and colour the limbs that are far away from the instructor's pose as red, else blue</li>
</ol>

Using this approach, the user can be able to scan through the video and find out areas that they need to improve on. Pictures help, so below is a flowchart describing the process:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic.png?raw=true)

This entire process will be implemented as a Flask API, which is why I recommended that you should preferably be comfortable with Flask. 

#### Architecture

Below is a diagram representing the architecture of the backend and its relationship to the frontend:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic2.png?raw=true)

We will create an endpoint that allows for a PUT request from the frontend in which the student and instructor video files will be received. RQ corresponds to a REDIS queue, where REDIS is an in-memory key-value storage system and is an excellent tool for creating job queues (serves users in a FIFO manner). This queue will receive the job on one of the available worker threads, and pose extraction will execute the series of steps as described in the overview. When the pose extraction module has finished a job, it will write the results to a file on disk. 

#### Backend file glossary
<ul>
  <li><b>./app/backend/app.py</b>
    <ul>
      <li>This corresponds to the Flask app, and this is where we will establish our endpoints for the frontend to make requests to</li>
    </ul>
  </li>
  <li><b>./app/backend/worker.py</b>
    <ul>
      <li>This is the worker thread that processes the jobs on the REDIS queue</li>
    </ul>
  </li>
  <li><b>./app/backend/util/pose_extraction.py</b>
    <ul>
      <li>This is a script that acts as a manager and is responsible for calling all necessary scripts and functions to perform the series of steps described in the overview</li>
    </ul>
  </li>
  <li><b>./app/backend/util/angles.py</b>
    <ul>
      <li>This corresponds to a series of functions that are necessary for extracting the angles between adjacent limbs, creating animations, and for computing the error between student and instructor</li>
    </ul>
  </li>
</ul>


### worker.py

This file establishes a connection with the REDIS server so that the worker can work work on a separate thread from the thread that the Flask app is on. Without this thread, requests made to the Flask API would be blocked. 

```python
import os

import redis
from loguru import logger
from rq import Worker, Queue, Connection

listen = ['high', 'default', 'low']

redis_url = 'redis://localhost:6379' 
logger.info('REDIS URL {}'.format(redis_url))
logger.info('In directory {}'.format(os.getcwd()))

conn = redis.from_url(redis_url)

if __name__ == '__main__':
  logger.info('__main__')
  with Connection(conn):
    worker = Worker(map(Queue, listen))
    logger.info(f'worker: {worker}')
    worker.work()
```

### app.py

Below are the imports and the decorators and headers for each one of the endpoints ('...' just corresponds to "we will fill this in"):

```python
import os
import json

from flask import (
    Flask, request, send_from_directory
)
from flask_cors import CORS
from loguru import logger
from rq import Queue
from rq.job import Job
from worker import conn

q = Queue(connection=conn)

app = Flask(__name__)
CORS(app)  # Cross-origin resource sharing (React app using different port than Flask app)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    ...

@app.route('/get_overlaps', methods=["GET"])
def get_overlaps():
    ...
```

#### upload_video

```python
@app.route('/upload_video', methods=['POST'])
def upload_video():
    video            = request.files
    instructor       = video['instructor']
    student          = video['student']
    instructor_fname = instructor.filename
    student_fname    = student.filename

    logger.info('Received {} & {}'.format(instructor_fname, student_fname))
    logger.info('In directory {}'.format(os.getcwd()))

    with open('../../videos/input/' + instructor_fname, 'wb') as f:
        f.write(instructor.read())
    
    with open('../../videos/input/' + student_fname, 'wb') as f:
        f.write(student.read())

    one_week = 60 * 60 * 24 * 7

    q.enqueue(
        'util.pose_extraction.get_error',
        args=(instructor_fname, student_fname),
        job_timeout=one_week
    )
    
    return {
        'status': 200,
        'mimetype': 'application/json'
    }
```

When the frontend makes a PUT request to 'upload_video', it sends instructor and student raw video files. rq then creates a new job by invoking the pose extraction.get_error utility function. Since jobs cannot have raw video passed in as a parameter, the files first need to be written to disk, and then the job can re-read the files later. The 'job_timeout' argument of the q.enqueue method represents the maximum possible time in seconds that the backend has to execute the job. Additionally, a separate worker thread is responsible for executing the job, thus this function can return while the job is processed asynchronously. This allows the Flask app to avoid getting blocked by requests. 

#### get_overlaps

```python
@app.route('/get_overlaps', methods=["GET"])
def get_overlaps():
    logger.info('Getting overlap data')

    files = ['/videos/{}'.format(f) for f in os.listdir('../frontend/public/videos') if '.mp4' in f]
    
    logger.info(files)
    return {
        'status': 200,
        'mimetype': 'application/json',
        'files': json.dumps(files)
    }
```

This endpoint is responsible for fetching the result videos from the server (when the frontend makes a new http request). 

### pose_extraction.py

Now that the Flask API has been created, the next step is to create a script that executes the process as described in the overview. I will first show the script in its entirety, then I will break it up into chunks that describe each major step.

```python
import os
import sys
import subprocess

import numpy as np

import torch
from loguru import logger

cur_dir = os.getcwd().split('/')[-1]
if cur_dir == 'util':
    import angles 
elif cur_dir == 'backend':
    import util.angles as angles
else:
    raise Exception('Not in proper directory')

def get_error(instructor, student):

    logger.info('Beginning process with instructor {} and student {}'.format(instructor, student))

    cur_dir = os.getcwd()

    main_dir = '../../'
    os.chdir(main_dir)

    logger.info('Beginning instructor inference')
    try:
        subprocess.run(['python3', 'setup/full_inference.py', '--input', instructor, '--output', 
                        'out_instructor.mp4', '--joints', 'joints-{}.npy'.format(instructor.split('.')[0])], check=True)
    except subprocess.CalledProcessError as e:
        logger.info(e.output)
        sys.exit(1)
    logger.info('Finished instructor inference')

    logger.info('Beginning student inference')
    try:
        subprocess.run(['python3', 'setup/full_inference.py', '--input', student, '--output', 
                        'out_student.mp4', '--joints', 'joints-{}.npy'.format(student.split('.')[0])], check=True)
    except subprocess.CalledProcessError as e:
        logger.info(e.output)
        sys.exit(1)
    logger.info('Finished student inference')

    instructor_pose = np.load('./joints/joints-{}.npy'.format(instructor.split('.')[0]))
    student_pose    = np.load('./joints/joints-{}.npy'.format(student.split('.')[0]))

    # Trim longer video to the length of the shorter video
    min_frames = min(len(instructor_pose), len(student_pose))
    if len(instructor_pose) > min_frames:
        instructor_pose = instructor_pose[:min_frames]
    if len(student_pose) > min_frames:
        student_pose = student_pose[:min_frames]

    instructor_pose_tensor = torch.from_numpy(instructor_pose)
    student_pose_tensor    = torch.from_numpy(student_pose)
    
    animation_directory = os.getcwd()
    # Run angle comparison code
    os.chdir(cur_dir)
    angles_between = angles.ang_comp(instructor_pose_tensor, student_pose_tensor, round_tensor=True)
    error = angles.error(angles_between)
    logger.info('Error {}'.format(error))

    os.chdir(animation_directory)
    # Get overlap between student and instructor
    ani, writer = angles.overlap_animation(instructor_pose, student_pose, error)
    instructor = instructor.split('.mp4')[0]
    student    = student.split('.mp4')[0]
    ani.save('./app/frontend/public/videos/{}-{}.mp4'.format(instructor, student), writer=writer)

    return error
```
 
#### Required imports, and running pose extraction on the student and instructor
```python
import os
import sys
import subprocess

import numpy as np

import torch
from loguru import logger

cur_dir = os.getcwd().split('/')[-1]
if cur_dir == 'util':
    import angles 
elif cur_dir == 'backend':
    import util.angles as angles
else:
    raise Exception('Not in proper directory')
    
def get_error(instructor, student):

    logger.info('Beginning process with instructor {} and student {}'.format(instructor, student))

    cur_dir = os.getcwd()

    main_dir = '../../'
    os.chdir(main_dir)

    logger.info('Beginning instructor inference')
    try:
        subprocess.run(['python3', 'setup/full_inference.py', '--input', instructor, '--output', 
                        'out_instructor.mp4', '--joints', 'joints-{}.npy'.format(instructor.split('.')[0])], check=True)
    except subprocess.CalledProcessError as e:
        logger.info(e.output)
        sys.exit(1)
    logger.info('Finished instructor inference')

    logger.info('Beginning student inference')
    try:
        subprocess.run(['python3', 'setup/full_inference.py', '--input', student, '--output', 
                        'out_student.mp4', '--joints', 'joints-{}.npy'.format(student.split('.')[0])], check=True)
    except subprocess.CalledProcessError as e:
        logger.info(e.output)
        sys.exit(1)
    logger.info('Finished student inference')
    ...
```

This block of code performs 3D pose estimation for both the instructor and student.

#### Creating an animation of the overlap between instructor and student, and returning the error between their poses
```python
    ...
    instructor_pose = np.load('./joints/joints-{}.npy'.format(instructor.split('.')[0]))
    student_pose    = np.load('./joints/joints-{}.npy'.format(student.split('.')[0]))
    
    # Trim longer video to the length of the shorter video
    min_frames = min(len(instructor_pose), len(student_pose))
    if len(instructor_pose) > min_frames:
        instructor_pose = instructor_pose[:min_frames]
    if len(student_pose) > min_frames:
        student_pose = student_pose[:min_frames]

    instructor_pose_tensor = torch.from_numpy(instructor_pose)
    student_pose_tensor    = torch.from_numpy(student_pose)
    
    animation_directory = os.getcwd()
    # Run angle comparison code
    os.chdir(cur_dir)
    angles_between = angles.ang_comp(instructor_pose_tensor, student_pose_tensor, round_tensor=True)
    error = angles.error(angles_between)
    logger.info('Error {}'.format(error))

    os.chdir(animation_directory)
    # Get overlap between student and instructor
    ani, writer = angles.overlap_animation(instructor_pose, student_pose, error)
    instructor = instructor.split('.mp4')[0]
    student    = student.split('.mp4')[0]
    ani.save('./app/frontend/public/videos/{}-{}.mp4'.format(instructor, student), writer=writer)

    return error
```

Once the 3D poses have been extracted, the longer of the two arrays is cropped to the size of the smaller array, and then the error calculation which we will go into more depth later is performed. Finally, an animation is created which shows the overlap between student and instructor, as well as the overall error for a particular frame (smoothed out across multiple frames).

#### full_inference.py

The main workhorse for this task is the VideoPose3D library <sup><a href='#ref1'>1</a></sup>. This library will be doing all of the 3D pose estimation for us, and we will be using pre-trained models. There are step by step instructions in https://github.com/facebookresearch/VideoPose3D/blob/master/INFERENCE.md on how to run VideoPose3D on a sample video, however the process needs to be automated for this app. 

This script is broken into four components, in order:

<ol>
  <li>Imports, loading of files, and exception handling</li>
  <li>2D keypoint detection using Detectron2<sup><a href='#ref2'>2</a></sup></li>
  <li>Creation of dataset that will be used in 3D keypoint detection</li>
  <li>3D keypoint detection</li>
</ol>

##### Parsing video arguments and exception handling
```python
# Inference script
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
...
```

This part of the script reads in the video file (the full path doesn't need to be specified, as they are assumed to be ./videos/input/) as well as the name of the output joints file.

##### Keypoint detection
```python
# Keypoint detection
os.chdir('inference')
logger.info('Beginning keypoint detection in directory {}'.format(os.getcwd()))
try:
    subprocess.run(['python3', 'infer_video_d2.py', '--cfg', 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml', 
                    '--output-dir', '../npz', '--image-ext', ext, '../../videos/input/{}'.format(input_file)], check=True)
except subprocess.CalledProcessError as e:
    logger.info(e.output)
    sys.exit(1)
...
```

This part of the script runs the 'infer_video_d2' script, which takes in the input file and performs 2D keypoint detection.

#### Dataset preparation
```python
# Dataset preparation
os.chdir('../data')
logger.info('Beginning 2D dataset preparation')
try:
    subprocess.run(['python3', 'prepare_data_2d_custom.py', '-i', '../npz', '-o', 'myvideos'], check=True)
except subprocess.CalledProcessError as e:
    logger.info(e.output)
    sys.exit(1)
...
```

This bit of code creates a custom dataset object that the library will use later for estimation 3D points.

#### 3D pose estimation
```python
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
```

Once complete, a numpy array of 3D keypoints across all frames is created and saved to a file. 

The array will be of size num_framesx17x3, since we have a matrix of keypoints for each frame, there are 17 joints that are being detected, and we are detecting 3D coordinates, hence a 3 at the end.

Here is a rough diagram that I made outlining the 3D coordinates being detected in each frame, indexed by the joints 0-based index location in the array.

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/video_pose_coordinates.png?raw=true)


### angles.py
Now that we have the 3D keypoints across all frames, we can create a module for correcting the student's pose in reference to an instructor. This will consist of finding the angles between adjacent limbs for the instructor and student, and then getting the sum of absolute angular differences to create an error vector for a frame.

Recall from linear algebra what it means to subtract two vectors:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/vector_subtraction.png?raw=true)

Therefore, using this same reasoning, to get the vector for a limb, we can just subtract two joint vectors:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/limb_vector.png?raw=true)

Then, to get the cosine of the angle between two adjacent limbs, we can use the following formula:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/cos_angle.png?raw=true)

By taking the arccosine of the result, we have the angle between two adjacent limbs.

Below is the entire code for this file:

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.signal as signal
import numpy as np

import torch
from pycpd import RigidRegistration

def ang_comp(reference, student, round_tensor=False):
    # Get all joint pair angles, frames x number of joint pairs

    adjacent_limb_map = [
                          [[0, 1],  [1, 2], [2, 3]],     # Right leg
                          [[0, 4],  [4, 5], [5, 6]],     # Left leg
                          [[0, 7],  [7, 8]],             # Spine
                          [[8, 14], [14, 15], [15, 16]], # Right arm
                          [[8, 11], [11, 12], [12, 13]], # Left arm
                          [[8, 9],  [9, 10]]             # Neck
                        ]
    
    adjacent_limbs_ref = []
    adjacent_limbs_stu = []
    num_frames = len(reference)

    def update_adjacent_limbs(person, adj, limb_id):
        for adj_limb_id in range(len(adjacent_limb_map[limb_id]) - 1):
            joint1a, joint1b = adjacent_limb_map[limb_id][adj_limb_id]
            joint2a, joint2b = adjacent_limb_map[limb_id][adj_limb_id + 1]
            
            limb1_vector = person[joint1a] - person[joint1b]  # Difference vector between two joints
            limb2_vector = person[joint2a] - person[joint2b]
            
            # Normalize the vectors
            limb1_vector = torch.div(limb1_vector, torch.norm(limb1_vector)).unsqueeze(0)
            limb2_vector = torch.div(limb2_vector, torch.norm(limb2_vector)).unsqueeze(0)
            
            adj.append(torch.Tensor(torch.cat([limb1_vector, limb2_vector], dim=0)).unsqueeze(0))

    for idx in range(num_frames):
        frame_reference = reference[idx]
        frame_student   = student[idx]
        for limb_id in range(len(adjacent_limb_map)):
            update_adjacent_limbs(frame_reference, adjacent_limbs_ref, limb_id)
            update_adjacent_limbs(frame_student, adjacent_limbs_stu, limb_id)
        
    adjacent_limbs_ref = torch.cat(adjacent_limbs_ref, dim=0)
    adjacent_limbs_stu = torch.cat(adjacent_limbs_stu, dim=0)

    # Get angles between adjacent limbs, each of the below tensors are of shape (num_frames x 10), aka scalars
    adjacent_limbs_ref = torch.bmm(adjacent_limbs_ref[:, 0:1, :], adjacent_limbs_ref[:, 1, :].unsqueeze(-1))
    adjacent_limbs_stu = torch.bmm(adjacent_limbs_stu[:, 0:1, :], adjacent_limbs_stu[:, 1, :].unsqueeze(-1))
    
    # Get absolute difference between instructor and student angles in degrees 
    # 57.296 * radians converts units to degrees
    absolute_diffs = torch.abs((57.296*(adjacent_limbs_ref - adjacent_limbs_stu))).reshape(num_frames, 10)
    return absolute_diffs.sum(dim=1)

def overlap_animation(reference, student, error):

    # Point set registration of reference and student
    transformed_student = []
    for idx in range(len(reference)):
        rt = RigidRegistration(X=reference[idx], Y=student[idx])
        rt.register()
        rt.transform_point_cloud()
        transformed_student.append(np.expand_dims(rt.TY, axis=0))

    student = np.concatenate(transformed_student, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    error_text = ax.text2D(1, 1, 'Error: 0', transform=ax.transAxes)
    
    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax.plot3D([], [], [], c='b') for _ in range(16)]
    stu_limbs = [ax.plot3D([], [], [], c='r') for _ in range(16)]
        
    limb_map = [
                [0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]
        
    def update_animation(idx):
        ref_frame = reference[idx]
        stu_frame = student[idx]
        
        for i in range(len(limb_map)):
            ref_limbs[i][0].set_data(ref_frame[limb_map[i], :2].T)
            ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 2])
            
            stu_limbs[i][0].set_data(stu_frame[limb_map[i], :2].T)
            stu_limbs[i][0].set_3d_properties(stu_frame[limb_map[i], 2])

        if idx < len(error):
            error_text.set_text('Error: {}'.format(int(error[idx])))
        
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    return ani, writer

def error(angle_tensor, window_sz=15):

    rolling_average = np.convolve(angle_tensor, np.ones(window_sz,)) / window_sz
    max_error = rolling_average.max()
    min_error = rolling_average.min()

    if max_error != min_error:
        rolling_average = (rolling_average - min_error) / (rolling_average - min_error)  # Normalize error between 0 and 1

    return rolling_average
```

This code will be described in detail over the next few sections.

#### Required imports

```python
# Angles extraction script
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import scipy.signal as signal
import numpy as np

import torch
from pycpd import RigidRegistration
...
```

#### Angle between
```python
def ang_comp(reference, student, round_tensor=False):
    # Get all joint pair angles, frames x number of joint pairs

    adjacent_limb_map = [
                          [[0, 1],  [1, 2], [2, 3]],     # Right leg
                          [[0, 4],  [4, 5], [5, 6]],     # Left leg
                          [[0, 7],  [7, 8]],             # Spine
                          [[8, 14], [14, 15], [15, 16]], # Right arm
                          [[8, 11], [11, 12], [12, 13]], # Left arm
                          [[8, 9],  [9, 10]]             # Neck
                        ]
    
    adjacent_limbs_ref = []
    adjacent_limbs_stu = []
    num_frames = len(reference)

    def update_adjacent_limbs(person, adj, limb_id):
        for adj_limb_id in range(len(adjacent_limb_map[limb_id]) - 1):
            joint1a, joint1b = adjacent_limb_map[limb_id][adj_limb_id]
            joint2a, joint2b = adjacent_limb_map[limb_id][adj_limb_id + 1]
            
            limb1_vector = person[joint1a] - person[joint1b]  # Difference vector between two joints
            limb2_vector = person[joint2a] - person[joint2b]
            
            # Normalize the vectors
            limb1_vector = torch.div(limb1_vector, torch.norm(limb1_vector)).unsqueeze(0)
            limb2_vector = torch.div(limb2_vector, torch.norm(limb2_vector)).unsqueeze(0)
            
            adj.append(torch.Tensor(torch.cat([limb1_vector, limb2_vector], dim=0)).unsqueeze(0))

    for idx in range(num_frames):
        frame_reference = reference[idx]
        frame_student   = student[idx]
        for limb_id in range(len(adjacent_limb_map)):
            update_adjacent_limbs(frame_reference, adjacent_limbs_ref, limb_id)
            update_adjacent_limbs(frame_student, adjacent_limbs_stu, limb_id)
        
    adjacent_limbs_ref = torch.cat(adjacent_limbs_ref, dim=0)
    adjacent_limbs_stu = torch.cat(adjacent_limbs_stu, dim=0)

    # Get angles between adjacent limbs, each of the below tensors are of shape (num_frames x 10), aka scalars
    adjacent_limbs_ref = torch.bmm(adjacent_limbs_ref[:, :1, :], adjacent_limbs_ref[:, 1, :].unsqueeze(-1))
    adjacent_limbs_stu = torch.bmm(adjacent_limbs_stu[:, :1, :], adjacent_limbs_stu[:, 1, :].unsqueeze(-1))
    
    # Get absolute difference between instructor and student angles 
    absolute_diffs = torch.abs(adjacent_limbs_ref - adjacent_limbs_stu).reshape(num_frames, 10)
    return absolute_diffs.sum(dim=1)
...
```

Given a numpy array of dimension num_frames x 17 x 3, computes the angles between all adjacent limbs for the student and instructor, and then computes the sum of absolute angular differences between them. 

The adjacent limb map contains all lists of adjacent limbs. Let's examine the first sublist in this list: \[\[0, 1], \[1, 2], \[2, 3]]. Each number corresponds to a joint as displayed in figure INSERT_FIGURE_NUMBER_HERE, and an entry like \[0, 1] corresponds to the directed limb that starts at joint 0 and goes to joint 1. These lists are organized in such a way that any list of size two directly beside each other (ie. \[0, 1] and \[1, 2]) are actually adjacent limbs. Using this list, to get the vector representation of a limb, we compute the difference vector between a pair of joints just as described in figure INSERT_FIGURE_NUMBER_HERE. And then, to get adjacent limbs, we need to get the limbs corresponding to sublists that are right beside each other. 

For example, in the nested function 'update_adjacent_limbs', the above paragraph corresponds to this piece of code:

```python
    def update_adjacent_limbs(person, adj, limb_id):
        for adj_limb_id in range(len(adjacent_limb_map[limb_id]) - 1):
            joint1a, joint1b = adjacent_limb_map[limb_id][adj_limb_id]
            joint2a, joint2b = adjacent_limb_map[limb_id][adj_limb_id + 1]
            
            limb1_vector = person[joint1a] - person[joint1b]  # Difference vector between two joints
            limb2_vector = person[joint2a] - person[joint2b]
            ...
```

Person corresponds to a an array of dimension 17 x 3 (aka, one frame of video). 'limb_id' corresponds to the index that adjacent_limb_map is accessed by, and adj_limb_id corresponds to the sub-sub list of adjacent_limb map. Once we have the joints for each adjacent limb, we just subtract them to get the actual adjacent limb vectors. 

Then, we normalize the size of each limb and append to a list:

```python
           ...
           # Normalize the vectors
            limb1_vector = torch.div(limb1_vector, torch.norm(limb1_vector)).unsqueeze(0)
            limb2_vector = torch.div(limb2_vector, torch.norm(limb2_vector)).unsqueeze(0)
            
            adj.append(torch.Tensor(torch.cat([limb1_vector, limb2_vector], dim=0)).unsqueeze(0))
            ...
```
 
'torch.norm', when given a vector, just returns the magnitude of the vector. By dividing a vector by its magnitude, it then has unit length. 'torch.cat' is essentially the tensor version of concatenation. Given a particular tensor dimension and a list of tensors, it concatenates the tensors across that particular dimension.

The next bit of code just gets the adjacent limbs across all frames for the student and instructor:
 
```python
    ...
    for idx in range(num_frames):
        frame_reference = reference[idx]
        frame_student   = student[idx]
        for limb_id in range(len(adjacent_limb_map)):
            update_adjacent_limbs(frame_reference, adjacent_limbs_ref, limb_id)
            update_adjacent_limbs(frame_student, adjacent_limbs_stu, limb_id)
   ...
```

Finally, once we have all adjacent limbs, we can compute the angles between them and get the sum of absolute angular differences between instructor and student:

```python
    ...
    adjacent_limbs_ref = torch.cat(adjacent_limbs_ref, dim=0)
    adjacent_limbs_stu = torch.cat(adjacent_limbs_stu, dim=0)
    
    # Get angles between adjacent limbs, each of the below tensors are of shape (num_frames x 10), aka scalars
    adjacent_limbs_ref = torch.bmm(adjacent_limbs_ref[:, 0:1, :], adjacent_limbs_ref[:, 1, :].unsqueeze(-1))
    adjacent_limbs_stu = torch.bmm(adjacent_limbs_stu[:, 0:1, :], adjacent_limbs_stu[:, 1, :].unsqueeze(-1))
    
    # Get absolute difference between instructor and student angles 
    absolute_diffs = torch.abs(adjacent_limbs_ref - adjacent_limbs_stu).reshape(num_frames, 10)
    ...
```

#### Creating an overlap animation
```python
def overlap_animation(reference, student, error):

    # Point set registration of reference and student
    transformed_student = []
    for idx in range(len(reference)):
        rt = RigidRegistration(X=reference[idx], Y=student[idx])
        rt.register()
        rt.transform_point_cloud()
        transformed_student.append(np.expand_dims(rt.TY, axis=0))
    
    student = np.concatenate(transformed_student, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    error_text = ax.text2D(1, 1, 'Error: 0', transform=ax.transAxes)
    # ax.axis('off')
    
    # There are 17 joints, therefore 16 limbs
    ref_limbs = [ax.plot3D([], [], []) for _ in range(16)]
    stu_limbs = [ax.plot3D([], [], []) for _ in range(16)]
        
    limb_map = [
                [0, 1],  [1, 2], [2, 3],     # Right leg
                [0, 4],  [4, 5], [5, 6],     # Left leg
                [0, 7],  [7, 8],             # Spine
                [8, 14], [14, 15], [15, 16], # Right arm
                [8, 11], [11, 12], [12, 13], # Left arm
                [8, 9],  [9, 10]             # Neck
               ]
        
    def update_animation(idx):
        ref_frame = reference[idx]
        stu_frame = student[idx]
        
        for i in range(len(limb_map)):
            ref_limbs[i][0].set_data(ref_frame[limb_map[i], :2].T)
            ref_limbs[i][0].set_3d_properties(ref_frame[limb_map[i], 2])
            
            stu_limbs[i][0].set_data(stu_frame[limb_map[i], :2].T)
            stu_limbs[i][0].set_3d_properties(stu_frame[limb_map[i], 2])

            if i < len(error):
                error_text.set_text('Error: {}'.format(error[i]))
        
    iterations = len(reference)
    ani = animation.FuncAnimation(fig, update_animation, iterations,
                                  interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    return ani, writer
```

This bit of code just creates an animation with the student and instructor coordinates aligned. I won't go into too much depth as to how this function works since it is matplotlib specific, but one important aspect to notice is this block of code:

```python
    ...
    # Point set registration of reference and student
    transformed_student = []
    for idx in range(len(reference)):
        rt = RigidRegistration(X=reference[idx], Y=student[idx])
        rt.register()
        rt.transform_point_cloud()
        transformed_student.append(np.expand_dims(rt.TY, axis=0))
    
    student = np.concatenate(transformed_student, axis=0)
    ...
```
What this portion does is something called rigid registration through an expectation maximization algorithm using the pycpd library<sup><a href='#ref3'>3</a></sup>. In short, it finds the best possible match between the student and instructor without scaling the student coordinates or changing the angles between the student's limbs in the process. 

#### Getting required angles

This function returns an animation object which overlays the student and instructor coordinates, as well as a writer to write to disk.

```python
def error(angle_tensor, window_sz=15):
    rolling_average = np.convolve(error, np.ones(window_sz,)) / window_sz
    return rolling_average
```

Given the difference in error in adjacent angles, computes a rolling average across all frames to remove any anomalies. 

## Building frontend 
### Before getting to React...

Ensure to copy the code contents of the following links into their respective files:

<ul>
  <li>https://github.com/DrJessop/yoga-pose/blob/staging/app/frontend/src/App.css into App.css</li>
  <li>https://github.com/DrJessop/yoga-pose/blob/staging/app/frontend/src/serviceWorker.js into serviceWorker.js</li>
</ul>

### The index files
#### index.html
```HTML
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta
      name="description"
      content="Web site created using create-react-app"
    />
    <link rel="apple-touch-icon" href="%PUBLIC_URL%/logo192.png" />
    <link rel="manifest" href="%PUBLIC_URL%/manifest.json" />
    <title>Vinyasa</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
```

This is just a basic HTML file, but there is one important thing to look at: the <div id="root"></div> section. This <div> tag is what we will render our application in, as described in the next section.

#### index.js
```JSX
import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import App from './App';
import * as serviceWorker from './serviceWorker';

ReactDOM.render(<App />, document.getElementById("root"));

serviceWorker.unregister();
```

This file is responsible for rendering the application App that we will be building in the next section. 'document.getElementById("root")' looks for the HTML element that has the Id root (the div tag at the end of index.html) and will render the contents of the application in that tag.

### Overview of React and our frontend
React is a framework that makes it easy for anyone to build an interactive user interface, and was designed for single-page application development. The "atoms" of React are called <b>components</b>, which are isolated pieces of code that allow the UI and the code logic to be loosely coupled. JSX is a markup language that allows components (code logic rendering) and UI to be merged together in a way that feels super similar to writing pure HTML. Therefore, instead of a complete separation of concerns (UI from logic), the motivation behind JSX is that "rendering logic is inherently coupled with the UI". 

The frontend for this yoga app will have 6 components:

<ul>
  <li><b>App</b>
    <ul>
      <li>"Manager" of the frontend</li>
      <li>Renders a menu at the top of the screen with links to each of the pages, and when the user clicks on a link, renders the associated component</li>
      <li>Contains a state attribute that can be passed on with each of the components</li>
      <li>Fetches all processed videos if they exist</li>
    </ul>
  </li>
  <li><b>HomePage</b>
    <ul>
      <li>Displays an enticing video that shows the VideoPose3D technology in action</li>
      <li>Displays the instructions that the user needs to follow to process their videos</li>
    </ul>
  </li>
  <li><b>HomeInstructions</b>
    <ul>
      <li>Contains instructions to the user so they know how to use the website</li>
    </ul>
  </li>
  <li><b>UploadVid</b>
    <ul>
      <li>Contains two dropzones, one for uploading the student video, and one for uploading the instructor video</li>
      <li>Contains a submit button that will create a put request to the backend, and will create a card ID for the ProcessedVideos components</li>
    </ul>
  </li>
  <li><b>ProcessedVideos</b>
    <ul>
      <li>Given an array containing video ids, titles, dates, and video paths, will render cards containing the progress of the backend for that associated card as well as the processed video when it is complete.</li>
    </ul>
  </li>
</ul>

We will go over each of the components in order.

### App.js
Although it may seem overwhelming, I will display the entire component block first, and then discuss each major section of the code.

```JSX
import React, { Component } from 'react';
import wld from './images/wld.jpeg';
import { BrowserRouter as Router, Link, Route} from 'react-router-dom';
import HomePage from './Components/HomePage';
import Team from './Components/Team';
import UploadVid from './Components/upload/UploadVid';
import ProcessedVideos from './Components/ProcessedVideos';

import 'animate.css';
import './App.css';

const ids = ['team', 'videos', 'upload'];

class App extends Component {

    constructor() {
        super();
        this.state = {title: [], link: [], date: [], path: []};
    }

    active_color(menu_id) {
        ids.forEach(id => document.getElementById(id).style.color = 'black');
        document.getElementById(menu_id).style.color = 'green';
    }
    
    onLoadColor() {
        if (window.location.href.includes('meetTheTeam')) {
            document.getElementById('team').style.color = 'green';
        }
        else if (window.location.href.includes('processed_videos')) {
            document.getElementById('videos').style.color = 'green';
        }
        else if (window.location.href.includes('upload')) {
            document.getElementById('upload').style.color = 'green';
        }
    }

    updateCardsFromTitle(filePath) {
        var fileSplit = filePath.split('/');
        var fileName = fileSplit[fileSplit.length - 1];
        this.updateCards(fileName, '', 'Oct 12, 2020', filePath);
    }
    
    onLoadCards() {
        fetch('http://127.0.0.1:5000/get_overlaps')
          .then(response => response.json())
          .then(response => JSON.parse(response.files).forEach(file => this.updateCardsFromTitle(file)));
    }
    
    onLoad() {
        this.onLoadColor();
        this.onLoadCards();
    }

    componentDidMount() {
        window.addEventListener('load', this.onLoad());
    }

    updateCards(title, link, date, path) {
        var newtitle = this.state.title.concat(title);
        var newlink  = this.state.link.concat(link);
        var newdate  = this.state.date.concat(date);
        var newpath  = this.state.path.concat(path)
        this.setState({title: newtitle, link: newlink, date: newdate, path: newpath});
        console.log(this.state);
    }

    render() {
        return (
            <Router>
                <ul id='home-menu' className='home-menu'>
                    <li className='left'>
                        <Link id='logo' to='/' onClick={() => this.active_color('logo')}>
                            <img src={wld} className='wld'/>
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/processed_videos' id='videos' className='right-text' onClick={() => this.active_color('videos')} >
                            Your Videos
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/upload' id='upload' className='right-text' onClick={() => this.active_color('upload')}>
                            Upload
                        </Link>
                    </li>
                </ul>
            <Route exact path='/' component={HomePage} />
            <Route exact path='/processed_videos' component={() => <ProcessedVideos title={this.state.title} 
                                                                                    link={this.state.link}
                                                                                    date={this.state.date}
                                                                                    path={this.state.path}/>} />
            <Route exact path='/upload' component={() => <UploadVid updateCards={(title, link, date, path) => 
                                                                                    this.updateCards(title, link, date, path)} />}/>
            </Router>
        );
    }
}

export default App;
```

#### Imports
```JSX
import React, { Component } from 'react';
import wld from './images/wld.jpeg';
import { BrowserRouter as Router, Link, Route} from 'react-router-dom';
import HomePage from './Components/HomePage';
import Team from './Components/Team';
import UploadVid from './Components/upload/UploadVid';
import ProcessedVideos from './Components/ProcessedVideos';

import 'animate.css';
import './App.css';
```
As described in the frontend overview, the App component serves as a driver for each of the other components, so it should make sense that we need to import each of the HomePage, Team, UploadVid, and ProcessedVideos components so that we can render them at some point in the application. More importantly, you will notice that we imported React and Component. Importing React will allow the JSX code that we will be writing to be rendered, and Component will be the parent class of the App component we are building.

#### Class definition and constructor
```JSX
...
const ids = ['team', 'videos', 'upload'];

class App extends Component {

    constructor() {
        super();
        this.state = {title: [], link: [], date: [], path: []};
    }
```

So, now this is your first time seeing the creation of a React component. Although not the only way to create a component, we are using a class to represent the App component, and if you want a class to be a component, it must extend the Component class to be treated as a React component and to inherit the useful methods associated with components. App also has a state attribute containing an array for each of title, link, date, and path. We will be using this state to represent the user's processed videos. The ids array just contains ids for the components that App will render, and we will need them later.

#### On loading of App
```JSX
    ...
    active_color(menu_id) {
        ids.forEach(id => document.getElementById(id).style.color = 'black');
        document.getElementById(menu_id).style.color = 'green';
    }
    
    onLoadColor() {
        if (window.location.href.includes('meetTheTeam')) {
            document.getElementById('team').style.color = 'green';
        }
        else if (window.location.href.includes('processed_videos')) {
            document.getElementById('videos').style.color = 'green';
        }
        else if (window.location.href.includes('upload')) {
            document.getElementById('upload').style.color = 'green';
        }
    }

    updateCardsFromTitle(filePath) {
        var fileSplit = filePath.split('/');
        var fileName = fileSplit[fileSplit.length - 1];
        this.updateCards(fileName, '', 'Oct 12, 2020', filePath);
    }
    
    onLoadCards() {
        fetch('http://127.0.0.1:5000/get_overlaps')
          .then(response => response.json())
          .then(response => JSON.parse(response.files).forEach(file => this.updateCardsFromTitle(file)));
    }
    
    onLoad() {
        this.onLoadColor();
        this.onLoadCards();
    }
    
    componentDidMount() {
        window.addEventListener('load', this.onLoad());
    }
    ...
```

The componentDidMount method always gets called when a component is rendered (such as is the case when the client makes an http request). We overloaded this method with our own event listener that will trigger our specific methods when the component is rendered. More specially, the onLoad method loads the associated colours of the page links (we want the active page link to have a different colour than the other page links) and also fetches processed videos if they exist. updateCardsFromTitle is a method that given the path of a processed video creates the attributes for generating a video card with updateCards (described in the next section).

#### Updating the state
```JSX
    ...
    updateCards(title, link, date, path) {
        var newtitle = this.state.title.concat(title);
        var newlink  = this.state.link.concat(link);
        var newdate  = this.state.date.concat(date);
        var newpath  = this.state.path.concat(path)
        this.setState({title: newtitle, link: newlink, date: newdate, path: newpath});
        console.log(this.state);
    }
```

A very important attribute is the state attribute of a React component, because it has its own reserved special method called setState. Notice how in this method, we didn't try to change the state variable ourselves (as in push the title, link, date, and path onto the state). The reason for this is because if we were to do this, a re-rendering step would not be performed and we wouldn't see the results of the state change in the UI. The setState method not only changes the changes, but any UI that depends on the state is re-rendered. Therefore, if your React component is a class and has an attribute that when changed should change the UI, it should be part of the state object and should be changed using setState.

#### Render

```JSX
    ...
    render() {
        return (
            <Router>
                <ul id='home-menu' className='home-menu'>
                    <li className='left'>
                        <Link id='logo' to='/' onClick={() => this.active_color('logo')}>
                            <img src={wld} className='wld'/>
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/meetTheTeam' id='team' className='right-text' onClick={() => this.active_color('team')}>
                            Meet the team
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/processed_videos' id='videos' className='right-text' onClick={() => this.active_color('videos')} >
                            Your Videos
                        </Link>
                    </li>
                    <li className='right'>
                        <Link to='/upload' id='upload' className='right-text' onClick={() => this.active_color('upload')}>
                            Upload
                        </Link>
                    </li>
                </ul>
            <Route exact path='/' component={HomePage} />
            <Route exact path='/meetTheTeam' component={Team} />
            <Route exact path='/processed_videos' component={() => <ProcessedVideos title={this.state.title} 
                                                                                    link={this.state.link}
                                                                                    date={this.state.date}
                                                                                    path={this.state.path}/>} />
            <Route exact path='/upload' component={() => <UploadVid updateCards={(title, link, date, path) => 
                                                                                    this.updateCards(title, link, date, path)} />}/>
            </Router>
        );
```
The render method is the most important method of a React component. This is the method that will allow our UI to be rendered by the DOM. You might notice that the block that is being returned looks awfully similar to HTML, except that React components show up in the code. This is called JSX markup.

There are few particular areas that are important to examine: first, the entire JSX block is wrapped inside of a Router component. React routers are incredibly useful for single-page apps since they allow us to dynamically render React components when the URL path matches a particular string. Link components are similar to html links, except to use React routers, one also needs to use Link components. When a user clicks on particular Link component, the URL path changes, and you can read the bottom block <Route exact path=PATH component=COMPONENT /> as a conditional block where if the path matches the associated path signature, then a particlar component will be rendered. 

The next important part to notice is when the path matches '/processed_videos'. 

```JSX
             ...
             <Route exact path='/processed_videos' component={() => <ProcessedVideos title={this.state.title} 
                                                                                     link={this.state.link}
                                                                                     date={this.state.date}
                                                                                     path={this.state.path}/>} />
             ...
```

React components all have a "props" object. Basically, it is the data that is passed from one component to another. In one of the ProcessedVideos methods, if I were to call this.props.title, it would correspond to the title that was passed to it from a different component.

The final important part of this code to examine is the final Route:

```JSX
            ...
            <Route exact path='/upload' component={() => <UploadVid updateCards={(title, link, date, path) => 
                                                                                    this.updateCards(title, link, date, path)} />}/>
            </Router>
            ...
```

In the same way that we can pass in attributes through props, we can also pass in methods. In this case, the updateCards method supplied to props here corresponds to a lambda function that when called triggers the App's updateCards method to be invoked. 

### Home page
This component is what the user will first see when they land on the app. 

```JSX
import React from 'react';
import TrackVisibility from 'react-on-screen';

import HomeInstructions from './instructions/home-instructions';
import yoga_output from '../videos/yoga_output.mp4';

const HomePage = () => {
    return (
        <>
            <video id='yoga-vid' className='videos' muted autoPlay loop>
                <source src={yoga_output} type='video/mp4' />
            </video>
            <center><p className='vid-text'>Find your flow</p></center>
            <TrackVisibility className='instructions'>
                {({ isVisible }) => isVisible && <div className="animate__animated animate__fadeInUp"><HomeInstructions/></div>}
            </TrackVisibility>
        </>
    )
}

export default HomePage;
```

The logic is simple. We have a background video called yoga_output which is stored in the videos directory which will play automatically and loop, and we have some cheesy line that describes the app, written as "Find your flow" which will hover across the background video. Finally, we have a simple animation where descriptive text shows up on to the screen to describe the series of steps that the user needs to perform to use the app. This is also the first time seeing a React component that is not part of a class, and thus does not need to extend Component.

The video instructions are part of the next component.

### home_instructions.js

```JSX
import React from 'react';

const HomeInstructions = () => {
    return (
        <div className='home-instructions'>
            <h2 className='home-instructions-header'><center>Only 3 steps</center></h2>
            <ul>
                <li className='home-instructions-steps'>
                    Upload videos
                </li>
                <li className='home-instructions-steps'>
                    Wait
                </li>
                <li className='home-instructions-steps'>
                    Improve
                </li>
            </ul>
        </div>
    );
}

export default HomeInstructions;
```

This component just returns an unordered list of instructions for the user.

### UploadVid.js

This component is responsible for the UI of the video upload page as well as the logic behind what happens when a particular video is updated as well as when a video is submitted. We will be using the React Dropzone library for this component, which makes file uploading incredibly easy. 

The logic behind this component is simple:

<ul>
  <li>This component will have a state containing two attributes: f1 and f2, where f1 corresponds to the instructor file and f2 corresponds to the instructor file. They will originally have null values</li>
  <li>There will be two dropzones
    <ol>
      <li>The first will contain a dropzone for the instructor, and when a user drops a video in this zone, the f1 attribute will be updated and the filename will appear under the zone to notify the user that their video was succesfully dropped.</li>
      <li>The second will do the same thing as the first, except it will be for the student video</li>
    </ol>
  </li>
  <li>There will be a submit button and...
    <ol>
      <li>...if the user clicks on the submit button and at least one of the files is null, then text appears notifying the user that they need to upload both instructor and student videos.</li>
      <li>...if both files are present, then the component will make a PUT request to the Flask app. Once the frontend receives a successful response code (status 200), then this component will update the cards of the parent App component</li>
    </ol>
  </li>
</ul>

Below is the entire code:

```JSX
import React, {Component, useCallback, useMemo, useState} from 'react';

import Dropzone, {useDropzone} from 'react-dropzone';
import Button from 'react-bootstrap/Button';

/* CSS 3rd party imports */
import '../../../node_modules/bootstrap/dist/css/bootstrap.min.css';


...
class MyDropzone extends Component{

  constructor(props) {
    super();
    this.state = {f1: null, f2: null};
  }

  on_drop1 = (acceptedFile) => {
    this.setState({f1: acceptedFile[0], f2: this.state.f2});
    console.log(this.state);
    document.getElementById('file1_upload').innerHTML = acceptedFile[0].path;
  }

  on_drop2 = (acceptedFile) => {
    this.setState({f1: this.state.f1, f2: acceptedFile[0]});
    console.log(this.state);
    document.getElementById('file2_upload').innerHTML = acceptedFile[0].path;
  }

  successful_upload = () => {
    document.getElementById('file1_upload').innerHTML = '';
    document.getElementById('file2_upload').innerHTML = '';
    this.setState({f1: null, f2: null});
    document.getElementById('submit_message').innerHTML = `Upload successful. Go to the videos tab to see results 
                                                           or upload more videos here.`;
  }

  submit = function() {

    if (this.state.f1 === null || this.state.f2 === null) {
      if (this.state.f1 === null) {
        document.getElementById('file1_upload').innerHTML = 'You must submit an instructor video';
      }
      if (this.state.f2 === null) {
        document.getElementById('file2_upload').innerHTML = 'You must submit a student video';
      }
      document.getElementById('submit_message').innerHTML = '';
      return;
    }

    let instructor = this.state.f1;
    let student    = this.state.f2;
    let date       = new Date;

    const time_ext = date.getTime().toString() + '.mp4';
    let instructor_fname = 'student_' + time_ext;
    let student_fname    = 'instructor_' + time_ext;

    const form_data = new FormData();

    form_data.append('instructor', instructor, instructor_fname);
    form_data.append('student', student, student_fname);

    fetch('http://127.0.0.1:5000/upload_video', {
      method: 'POST',
      body: form_data
    }).then(response => response.json())
      .then(response => console.log(response))
      .then(this.successful_upload())
      .then(this.props.updateCards(instructor_fname + ' ' + student_fname, "in progress", date.getDate()));
  }

  render() {
    return (
      <div>
        <ol className='upload_instructions'>
            <li style={{paddingTop:'7%'}}>
              Drag and drop a video of your favourite instructor's class
              
              <Dropzone onDrop={this.on_drop1} accept='video/mp4' multiple={false}>
                {({getRootProps, getInputProps}) => (
                  <div {...getRootProps()} style={baseStyle}>
                    <input {...getInputProps()} />
                    Drop file here
                  </div>
                )}
              </Dropzone>
              <p id='file1_upload' />
            </li>
            <li>
              Drag and drop a video of a student following the instructor's class
              <Dropzone onDrop={this.on_drop2} accept='video/mp4' multiple={false}>
              {({getRootProps, getInputProps}) => (
                <div {...getRootProps()} style={baseStyle}>
                  <input {...getInputProps()} />
                  Drop file here
                </div>
              )}
            </Dropzone>
            <p id='file2_upload' />
            </li>
            <li>
              Click the submit button and we will then process your video
              <div>
                <Button id='submit_button' size='lg' onClick={this.submit.bind(this)}>
                  Submit
                </Button>
                <p id='submit_message' style={{paddingTop:'3%', color: 'black'}} />
              </div>
                
            </li>
        </ol>
      </div>
    );
  }
}

export default MyDropzone;
```

Let's break each step down...

#### Attributes and attribute setters
```JSX
class MyDropzone extends Component{

  constructor(props) {
    super();
    this.state = {f1: null, f2: null};
  }

  on_drop1 = (acceptedFile) => {
    this.setState({f1: acceptedFile[0], f2: this.state.f2});
    console.log(this.state);
    document.getElementById('file1_upload').innerHTML = acceptedFile[0].path;
  }

  on_drop2 = (acceptedFile) => {
    this.setState({f1: this.state.f1, f2: acceptedFile[0]});
    console.log(this.state);
    document.getElementById('file2_upload').innerHTML = acceptedFile[0].path;
  }
```

Just like in App.js, this class must extend Component. The state attribute contains two file attributes, f1 and f2, which are obviously null when this is rendered for the first time. Since we will have two dropzones, we need to have two different on_drop methods corresponding to which dropzone was accessed. When a file is dropped in the first dropzone, the first file of state has to change. Recall from the previous section that it's best practice and sometimes required to set the state using the setState method, so we will invoke it here.

```JSX
  ...
  successful_upload = () => {
    document.getElementById('file1_upload').innerHTML = '';
    document.getElementById('file2_upload').innerHTML = '';
    this.setState({f1: null, f2: null});
    document.getElementById('submit_message').innerHTML = `Upload successful. Go to the videos tab to see results 
                                                           or upload more videos here.`;
  }
  ...
```

This method is responsible for setting the state back to null after re-rendering.

#### Submitting
```JSX
  ...
  submit = function() {

    if (this.state.f1 === null || this.state.f2 === null) {
      if (this.state.f1 === null) {
        document.getElementById('file1_upload').innerHTML = 'You must submit an instructor video';
      }
      if (this.state.f2 === null) {
        document.getElementById('file2_upload').innerHTML = 'You must submit a student video';
      }
      document.getElementById('submit_message').innerHTML = '';
      return;
    }

    let instructor = this.state.f1;
    let student    = this.state.f2;
    let date       = new Date;

    const time_ext = date.getTime().toString() + '.mp4';
    let instructor_fname = 'student_' + time_ext;
    let student_fname    = 'instructor_' + time_ext;

    const form_data = new FormData();

    form_data.append('instructor', instructor, instructor_fname);
    form_data.append('student', student, student_fname);

    fetch('http://127.0.0.1:5000/upload_video', {
      method: 'POST',
      body: form_data
    }).then(response => response.json())
      .then(response => console.log(response))
      .then(this.successful_upload())
      .then(this.props.updateCards(instructor_fname + ' ' + student_fname, "in progress", date.getDate()));
  }
  ...
```

This method first checks whether at least one of the files is null, and displays a message to upload both an instructor and student video to the user if so. Else, a FormData object is created and is populated with the instructor and student raw files and filenames, and then submits a fetch request to the Flask API 'upload_video' endpoint. The fetch function is a typical javascript function, but if you are not familiar with it...

TALK ABOUT PROMISES.

### ProcessedVideos.js

This component is responsible for rendering all of the cards (which contain the processed videos) to the UI. The code is displayed below:

```JSX
import React from 'react';
import Card from 'react-bootstrap/Card';

import '../../node_modules/bootstrap/dist/css/bootstrap.min.css';

const ProcessedVideos = (props) => {

    var html = [];
    console.log(props);
    for (var idx = 0; idx < props.title.length; idx++) {
        html.push(
        <div id={idx} style={{padding:'7%'}}>
            <Card>
                <Card.Header>{props.title[idx]}</Card.Header>
                <Card.Body>
                    <blockquote className="blockquote mb-0">
                    <p>
                    <video muted controls className='videos'>
                        <source src={props.path[idx]} type='video/mp4' />
                    </video>
                    </p>
                    <footer className="blockquote-footer">
                        {props.date[idx]}
                    </footer>
                    </blockquote>
                </Card.Body>
            </Card>
        </div>);
    }

    return (
        <div>
            {html}
        </div>
    );
}

export default ProcessedVideos;
```

## Launching the application
Launching the app has to be done in several steps (I won't bother trying to dockerize the application in this tutorial):

In yoga-pose/app/frontend/, run

```sh
npm start
```

In a separate terminal, run

```sh
redis-server
```

and ensure that you see that it is running on port 6379.

Next, in a separate terminal, from yoga-pose/app/backend, run

```sh
python app.py
```

and then finally, in one more terminal, also in yoga-pose-app/backend, run

```sh
python worker.py
```

## What's next for Yoga Pose
<ul>
  <li>It would be incredibly useful to a student to get real-time verbal feedback from the AI, not just feedback by submitting videos.</li>
  <li>Student lag was not taken into consideration by the algorithm (student will always be a few seconds behind the instructor), and that would be useful to include to give better feedback to the student on their performance.</li>
  <li>For every joint, VideoPose3D gives a single point describing its location, but doesn't take direction into account (whether a joint flipped around for a particular pose or not, so a new model might have to take that into consideration</li>
  <li>Anomaly detection is also important (for example, if the instructor needs to brush hair out of their face, the student should not be penalized if they don't do the same thing</li>
  <li>It would be useful to segment the video by sequences so that the student can easily see their performance for a particular sequence</li>
</ul>

## References 
<ol>
  <li><h2 id='ref1'><a href="https://github.com/facebookresearch/VideoPose3D">Facebook Video Pose 3D</a></h2></li>
  <li><h2 id='ref2'><a href="https://github.com/facebookresearch/detectron2">Detectron2</a></h2></li>
  <li><h2 id='ref3'><a href="https://github.com/siavashk/pycpd">PYCPD</a></h2></li>
</ol>

## License
The MIT License (MIT)

Copyright (c) 2020 Andrew Grebenisan

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


