import os
import numpy as np

import torch
from loguru import logger

import util.angles

def get_error(instructor, student):

    logger.info('Beginning process with instructor {} and student {}'.format(instructor, student))

    cur_dir = os.getcwd()
    main_dir = '../../'
    os.chdir(main_dir)

    # Instructor pose estimation
    logger.info('Beginning instructor inference')
    os.system('python setup/full_inference.py --input {} --output out_instructor.mp4 --joints joints_instructor.npy'.format(instructor))

    # Student pose estimation
    logger.info('Beginning student inference')
    os.system('python setup/full_inference.py --input {} --output out_student.mp4 --joints joints_student.npy'.format(student))

    instructor_pose = torch.from_numpy(np.load('./joints/joints_instructor.npy'))
    student_pose    = torch.from_numpy(np.load('./joints/joints_student.npy'))

    # Run angle comparison code
    os.chdir(cur_dir)
    angles_between = angles.ang_comp(instructor_pose, student_pose, round_tensor=True)
    error = angles.error(angles_between)
    error = torch.mul(error, (error > 5))  # If error is less than 5 degrees, it becomes 0
    max_error = error.max()
    min_error = error.min()
    error = (error - min_error) / (max_error - min_error)  # Normalize error between 0 and 1
    logger.info('Error {}'.format(error))
    return error



