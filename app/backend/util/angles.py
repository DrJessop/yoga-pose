import torch
import pypcd

def angle_between(t1, t2, round_tensor=False):
    norm1   = torch.norm(t1, dim=2).unsqueeze(-1)
    norm2   = torch.norm(t2, dim=2).unsqueeze(-1)
    unit_t1 = torch.div(t1, norm1)
    unit_t2 = torch.transpose(torch.div(t1, norm2), 2, 1)
    angles  = torch.bmm(unit_t1, unit_t2).clamp(-1, 1).acos()
    
    if round_tensor:
        angles = torch.round(angles)
    
    return angles

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

def error(angle_tensor):
    absolute_error = angle_tensor.sum(dim=1).view(-1)
    return absolute_error
