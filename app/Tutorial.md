# Building a Yoga Assistant App with React, Flask, and Pytorch
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
      <li>How to share videos through Facebook messenger platform</li>
    </ol>
  </li>
</ol>

This is a beginner tutorial for <b>React</b> and <b>Pytorch</b>, however there are some pre-requisites. Prior to this tutorial, you should be comfortable with:

<ul>
  <li>Python3+</li>
  <li>Flask</li>
  <li>Javascript (optional)</li>
</ul>

## Getting Started
Before beginning, you need to install the following:
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

### Required python modules

The requirements file for all necessary Python modules can be found in in app/backend 

To install, in terminal, run

```sh
python3 -m pip install -r requirements.txt
```

### Trouble-shoot for detectron2
Problem with gcc and g++

### Directory structure
Below is a diagram of the directory structure of the entire project. Using this structure, create empty files with the following names, as the rest of the tutorial will be referring to these files.


## Building python backend 
### Architecture of backend
In this section, you will be building a Flask API backed By VideoPose3D which will provide the necessary functions to correct a student's pose based off of their instructor. Pose estimation refers to estimating joint key-points on a subject and connecting them together. In 3D pose estimation, the true depth of the joints is also estimated. By estimating the poses of the student and instructor, we can figure out what adjustments the student needs to make to match the pose of the instructor. Below is a diagram representing the architecture of the backend and its relationship to the frontend:

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic2.png?raw=true)

Gunicorn is a WSGI server that is the way in which the frontend will commute with the Flask API. RQ corresponds to a REDIS queue, where REDIS is an in-memory key-value storage system and is an excellent tool for creating job queues (serves users in a FIFO manner). The backend will receive a put request with two video files (student and instructor), afterwhich a REDIS queue will launch a job to be performed by a worker thread. 

### Worker thread
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

This file is necessary for establishing a separate worker thread for the flask app. When the client submits a request to 

### Flask API
In this part of the tutorial, the different components of the Flask app will be discussed.

Below are the top of the file and the decorators and headers for each one of the endpoints:

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

@app.route('/videos/overlaps/<path:path>')
def send_static(path):
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

When the frontend makes a PUT request to 'upload_video', it sends instructor and student raw video files. rq then creates a new job by invoking the pose extraction.get_error utility function. Since jobs cannot have raw video passsed in as a parameter, the files first need to be written to disk, and then the job can re-read the files later. The 'job_timeout' argument of the q.enqueue method represents the maximum possible time in seconds that the backend has to execute the job. Additionally, a separate worker thread is responsible for executing the job, thus this function can return while the job is processed asynchronously. This allows the Flask app to avoid getting blocked by requests. 

#### send_static

```python
@app.route('/videos/overlaps/<path:path>')
def send_static(path):
    logger.info(path)
    return send_from_directory('videos/overlaps', path)
```

This endpoint is responsible for serving static files. Since javascript is client-side, it has no way of reading in files on the server, but it can make a get request to be able to fetch video data from the server.

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
    ...
```

This part of the function performs the 'infer keypoints' in the above diagram on the instructor and student. The function is continued in the next block...

```python
    ...
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

This part of the function loads the keypoints from memory, creates a video of the best possible overlap between the instructor and student, and then runs the angle comparison code which returns a rolling average error vector between the instructor and student across all frames.

#### full_inference.py

The main workhorse for this task is the VideoPose3D library <sup><a href='#ref1'>1</a></sup>. In short, VideoPose3D performs 2D keypoint detection (with Detectron2 <sup><a href='#ref2'>2</a></sup>) across all frames of an input video, and then using temporal information between frames of 2D keypoints, does something called 'back-projection' which finds the most probable 3D pose given the input video.

/* TODO: Have full setup instructions at the very BEGINNING of the tutorial 
There is a script in the root folder of this repository called /setup/videopose_setup.py. This script will clone the VideoPose3D repository and install Detectron2. Afterwards, the Detectron2 model will have to be downloaded (https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin) and placed into /YogaPose3D/checkpoint. 
*/

There are step by step instructions in https://github.com/facebookresearch/VideoPose3D/blob/master/INFERENCE.md on how to run VideoPose3D on a sample video, however the process needs to be automated for this app. 

This script is broken into four components, in order:

<ol>
  <li>Imports, loading of files, and exception handling</li>
  <li>2D keypoint detection using Detectron2</li>
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

This part of the script runs the 'infer_video_d2' script, which is a part of VideoPose3D. It takes in the input file and performs 2D keypoint detection.

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

### angle extraction


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

```python
def angle_between(t1, t2, round_tensor=False):
    norm1   = torch.norm(t1, dim=2).unsqueeze(-1)
    norm2   = torch.norm(t2, dim=2).unsqueeze(-1)
    unit_t1 = torch.div(t1, norm1)
    unit_t2 = torch.transpose(torch.div(t1, norm2), 2, 1)

    eps = 1e-7
    cos_angles  = torch.bmm(unit_t1, unit_t2).clamp(-1 + eps, 1 - eps)
    angles = cos_angles.acos()
    print(cos_angles)
    
    if round_tensor:
        angles = torch.round(angles)
    
    return angles
...
```
```python
def ang_comp(reference, student, round_tensor=False):
    angles = angle_between(reference, student, round_tensor)
    
    pelvis_rhip  = angles[:, 0, 1].unsqueeze(1)
    rhip_rknee   = angles[:, 1, 2].unsqueeze(1)
    rknee_rankle = angles[:, 2, 3].unsqueeze(1)
    pelvis_lhip  = angles[:, 0, 4].unsqueeze(1)
    lhip_lknee   = angles[:, 4, 5].unsqueeze(1)
    lknee_lankle = angles[:, 5, 6].unsqueeze(1)
    pelvis_spine = angles[:, 0, 7].unsqueeze(1)

    angles = torch.cat([pelvis_rhip, rhip_rknee, rknee_rankle, 
                        pelvis_lhip, lhip_lknee, lknee_lankle, 
                        pelvis_spine], axis=1)

    return angles
...
```
```python
def overlap(reference, student):

    assert len(reference) == len(student)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.axis('off')
    ax.set_xlim3d(-1., 1.)
    ax.set_ylim3d(-1., 1.)
    ax.set_zlim3d(0.,  1.)

    scat = ax.scatter([], [], color='red', marker='o') 
    scat2 = ax.scatter([], [], color = 'blue', marker = 'o')

    iterations = len(reference)

    def update_animation(idx):

        ref_pts = reference[idx]
        student_pts = student[idx]

        scat._offsets3d = (ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2])
        scat2._offsets3d = (student_pts[:, 0], student_pts[:, 1], student_pts[:, 2])
    
    ani    = animation.FuncAnimation(fig, update_animation, iterations,
                                       interval=50, blit=False, repeat=True)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    return ani, writer
...
```
```python
def error(angle_tensor, window_sz=15):
    error = angle_tensor.sum(dim=1).view(-1)

    rolling_average = np.convolve(error, np.ones(window_sz,)) / window_sz
    max_error = rolling_average.max()
    min_error = rolling_average.min()

    if max_error != min_error:
        rolling_average = (rolling_average - min_error) / (rolling_average - min_error)  # Normalize error between 0 and 1

    return rolling_average
```

## Building frontend in React
Now that the workhorse of the app has been created, the next step is to make a frontend. The frontend should have a homepage, a page for uploading videos, and a page where you can see your processed videos.

### App.js
In a typical React app, there is a file called App.js that serves as a driver for the program. This file will allow for the rendering of each of the pages, as well as triggering specific methods when the website is loaded. 

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

    constructor() {
        super();
        this.state = {title: [], link: [], date: [], path: []};
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
    }
}

export default App;
```

### Home page

### 'Video upload' page
```JSX
import React, {Component, useCallback, useMemo, useState} from 'react';

import Dropzone, {useDropzone} from 'react-dropzone';
import Button from 'react-bootstrap/Button';

/* CSS 3rd party imports */
import '../../../node_modules/bootstrap/dist/css/bootstrap.min.css';


const baseStyle = {
  flex: 1,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'center',
  padding: '20px',
  marginLeft: '10%',
  marginRight: '13%',
  cursor: 'pointer',
  borderWidth: 2,
  borderRadius: 2,
  borderColor: '#eeeeee',
  borderStyle: 'dashed',
  backgroundColor: '#fafafa',
  color: '#bdbdbd',
  outline: 'none',
  transition: 'border .24s ease-in-out'
};

const activeStyle = {
  borderColor: '#2196f3'
};

const acceptStyle = {
  borderColor: '#00e676'
};

const rejectStyle = {
  borderColor: '#ff1744'
};

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

### 'Your videos' page
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

## What's Next 
- Real-time application
- Segmentation of video using yoga pose classification
- Rotation of limbs

## References 
<ol>
  <li><h2 id='ref1'>Reference 1</h2></li>
  <li><h2 id='ref2'>Reference 2</h2></li>
  <li><h2 id='ref3'>Reference 3</h2></li>
</ol>



