import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    return rolling_average
