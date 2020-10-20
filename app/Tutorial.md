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
In this section, we will be building a REST API backed By VideoPose3D. Pose estimation refers to estimating joint key-points on a subject and connecting them together. In 3D pose estimation, the true depth of the joints is also estimated. By estimating the poses of the student and instructor, we can figure out what adjustments the student needs to make to match the pose of the instructor. 
Below is a diagram representing the series of steps that need to be performed for student correction.

![alt text](https://github.com/DrJessop/yoga-pose/blob/staging/app/images/backend_schematic.png?raw=true)

The main workhorse for this task is the VideoPose3D library<sup><a href='ref1'>1</a></sup>. There is a script in the root folder of this repository called /setup/videopose_setup.py. This script will clone the VideoPose3D repository and install Detectron2<sup>2</sup>. 

### Flask API




This library has an 'in-the-wild' mode that allows users to experiment on their own sample videos. They describe a series of steps that must be undertaken to perform pose estimation, however we will not be expecting the end user to perform those tasks, so we instead need to automate the steps with a python script.

### infer.py
The goal of this piece of code is to take a video and write the 3D coordinates of the limbs across all frames to a .npy file. 

### Building utility functions
Now that we have built the code that extracts the 3D coordinates, we need to create a function that given two torch tensors, generates an error vector which represents the rolling average error across all frames. 

### angles.py

### Main worker

### Building Flask API

# Trouble-shoot for detectron2
Problem with gcc and g++

## Building frontend in React


## What's Next 

## References 
<ol>
  <li><h2 id='ref1'>Reference 1</h2></li>
</ol>



