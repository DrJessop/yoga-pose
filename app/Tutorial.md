# Building a Yoga Assistant App with React, Flask, and Pytorch
During the pandemic, a lot of fitness-related activities went fully online, including yoga. I'm sure many fellow yogis have felt the frustration of not getting feedback on their posture during this time. In this tutorial, you will be building an application that helps correct yoga poses using React, Flask, and Facebook's VideoPose3D. 

This tutorial will cover 

<ul>
  <li>Flask endpoints</li>
  <li>Redis job queues</li>
  <li>React routers for different parts of the homepage</li>
  <li>File uploading and pre-processing in React</li>
  <li>Flask SocketIO</li>
  <li>React cards</li>
</ul>

Prior to this tutorial, you should be comfortable with the syntax of

<ul>
  <li>Python3+</li>
  <li>Javascript</li>
</ul>

## Getting Started
Before even beginning, you need to install the following:
<ul>
  <li>node package manager (npm)</li>
  <li>python3+</li>
</ul>

### Installing npm on...

#### Mac
brew install node
#### Linux
#### Windows

### Installing python3+ on...

#### Mac
#### Linux
#### Windows

Additionally, there is a requirements file for all necessary Python modules in app/backend and for all necessary npm modules in app/frontened.

## Building python backend 
### Architecture of backend
In this section, we will be building a REST API backed By VideoPose3D. Pose estimation refers to estimating joint key-points on a subject and connecting them together. In 3D pose estimation, the true depth of the joints is also estimated. By estimating the poses of the student and instructor, we can figure out what adjustments the student needs to make to match the pose of the instructor. 
Below is a diagram representing the architecture of the backend and its relationship to the frontend.

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic2.png?raw=true)

Gunicorn is a WSGI server that is the method in which the frontend will commute with the Flask API. The backend will receive a put request with two video files (student and instructor), afterwhich a REDIS queue will launch a job to be performed by a worker thread. 

### Flask API
This section will describe creating the endpoints for the Flask API. There are three endpoints:

```python
@app.route('/upload_video', methods=['POST'])
def upload_video():
    ...

@app.route('/videos/overlaps/<path:path>')
def send_static(path):
    ...

@app.route('/get_overlaps', methods=["GET"])
def get_overlaps():
    ...
```

Each one of these endpoints will be broken down separately.

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

When the frontend makes a PUT request to 'upload_video', it sends instructor and student raw video files. rq then creates a new job by invoking the pose extraction.get_error utility function. Since jobs cannot have raw video passsed in as a parameter, the files first need to be written to disk, and then the job can re-read the files later.

```python
@app.route('/videos/overlaps/<path:path>')
def send_static(path):
    logger.info(path)
    return send_from_directory('videos/overlaps', path)
```

This endpoint is responsible for serving static files. Since javascript is client-side, it has no way of reading in files on the server, but it can make a get request to be able to fetch video data from the server.

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

This endpoint is responsible for fetching the result videos from the server.

### Pose extraction module

Now that the Flask API has been created, the next step is to create the 'pose_extraction' module that contains all the useful functions necessary for obtaining the poses of the instructor and student, returning the error between instructor and student, student correction, and video generation.

The 'pose_extraction' worker function executes the following steps:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic.png?raw=true)

More specifically, in code:

```python
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

    ani, writer = angles.overlap(instructor_pose, student_pose)
    instructor = instructor.split('.mp4')[0]
    student    = student.split('.mp4')[0]
    ani.save('./app/frontend/public/videos/{}-{}.mp4'.format(instructor, student), writer=writer)
    
    instructor_pose = torch.from_numpy(instructor_pose)
    student_pose    = torch.from_numpy(student_pose)
    
    # Run angle comparison code
    os.chdir(cur_dir)
    angles_between = angles.ang_comp(instructor_pose, student_pose, round_tensor=True)
    error = angles.error(angles_between)
    logger.info('Error {}'.format(error))

    # Create point set 'registration'
    return error
```

#### Worker

The main workhorse for this task is the VideoPose3D library <sup><a href='#ref1'>1</a></sup>. In short, VideoPose3D performs 2D keypoint detection (with Detectron2 <sup><a href='#ref2'>2</a></sup>) across all frames of an input video, and then using temporal information between frames of 2D keypoints, does something called 'back-projection which finds the most probable 3D pose given the input video.

There is a script in the root folder of this repository called /setup/videopose_setup.py. This script will clone the VideoPose3D repository and install Detectron2. Afterwards, the Detectron2 model will have to be downloaded (https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and placed into /YogaPose3D/checkpoint. 

There are step by step instructions in https://github.com/facebookresearch/VideoPose3D/blob/master/INFERENCE.md on how to run VideoPose3D on a sample video, however we need to automate the process using a python script. The script can be seen in https://github.com/DrJessop/yoga-pose/blob/staging/setup/full_inference.py. 

The script is broken into 3 parts.

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
```

```python
# Angles extraction script

```

```python
# Worker script

```


# Trouble-shoot for detectron2
Problem with gcc and g++

## Building frontend in React


## What's Next 

## References 
<ol>
  <li><h2 id='ref1'>Reference 1</h2></li>
  <li><h2 id='ref2'>Reference 2</h2></li>
</ol>



