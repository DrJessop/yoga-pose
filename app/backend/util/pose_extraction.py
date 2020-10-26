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