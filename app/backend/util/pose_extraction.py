import os
import sys
import subprocess

import numpy as np

import torch
from loguru import logger
from pycpd import RigidRegistration

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

    instructor_pose = torch.from_numpy(np.load('./joints/joints-{}.npy'.format(instructor.split('.')[0])))
    student_pose    = torch.from_numpy(np.load('./joints/joints-{}.npy'.format(student.split('.')[0])))

    '''
    # Run angle comparison code
    os.chdir(cur_dir)
    angles_between = angles.ang_comp(instructor_pose, student_pose, round_tensor=True)
    error = angles.error(angles_between)

    # error = torch.mul(error, (error > 5))  # If error is less than 5 degrees, it becomes 0
    max_error = error.max()
    min_error = error.min()
    error = (error - min_error) / (max_error - min_error)  # Normalize error between 0 and 1
    logger.info('Error {}'.format(error))

    # Create point set 'registration'
    return error
    '''
    



