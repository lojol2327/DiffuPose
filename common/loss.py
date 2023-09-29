import torch
import numpy as np

import torch.nn.functional as F
from common.args import parse_args
from common.camera import get_uvd2xyz
from einops import rearrange, reduce

args = parse_args()

def get_w(weighted=True):
    if weighted:
        if args.dataset == 'h36m':
            if args.keypoints == 'hr':
                w = [1., # 0: pelvis
                    1., # 1: R_Hip
                    2., # 2: R_Knee
                    4., # 3: R_Ankle
                    1, # 4: L_Hip
                    2., # 5: L_Knee
                    4., # 6: L_Ankle
                    1., # 7: Spine
                    1., # 8: Torso
                    1., # 10: Head
                    1., # 11: L_Shoulder
                    2., # 12: L_Elbow
                    4., # 13: L_Wrist
                    1., # 14: R_Shoulder
                    2., # 15: R_Elbow
                    4.] # 16: R_Wrist
            else:
                w = [1., # 0: pelvis
                    1., # 1: R_Hip
                    2., # 2: R_Knee
                    4., # 3: R_Ankle
                    1, # 4: L_Hip
                    2., # 5: L_Knee
                    4., # 6: L_Ankle
                    1., # 7: Spine
                    1., # 8: Torso
                    2., # 9: Nose/Neck
                    1., # 10: Head
                    1., # 11: L_Shoulder
                    2., # 12: L_Elbow
                    4., # 13: L_Wrist
                    1., # 14: R_Shoulder
                    2., # 15: R_Elbow
                    4.] # 16: R_Wrist
        elif args.dataset == 'humaneva15':
            w = [1.0,
                 1.0,
                 2.5,
                 2.5,
                 1.0,
                 2.5,
                 2.5,
                 1.0,
                 1.5,
                 1.5,
                 4.0,
                 4.0,
                 1.5,
                 4,0,
                 4.0]
    else:
        w = [1.0] * 17
    
    return torch.tensor(w).reshape(1, 1, -1, 1)

def mean_velocity_error_train(predicted, target, axis=0):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = torch.diff(predicted, dim=axis)
    velocity_target = torch.diff(target, dim=axis)

    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=len(target.shape)-1))

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
    
def weighted_mpjpe(predicted, target, w):
    """
    Weighted mean per-joint position error (i.e. mean Euclidean distance)
    """
    assert predicted.shape == target.shape
    assert w.shape[0] == predicted.shape[0]
    return torch.mean(w * torch.norm(predicted - target, dim=len(target.shape)-1))

def p_mpjpe(predicted, target):
    assert predicted.shape == target.shape
    
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    X0 = torch.from_numpy(X0.transpose(0, 2, 1))
    Y0 = torch.from_numpy(Y0)
    H = (X0 @ Y0).numpy()
    # H = np.matmul(X0.transpose(0, 2, 1), Y0)
    U, s, Vt = np.linalg.svd(H)
    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

    a = tr * normX / normY # Scale
    t = muX - a*np.matmul(muY, R) # Translation
    
    # Perform rigid transformation on the input
    predicted_aligned = a*np.matmul(predicted, R) + t
    
    # Return MPJPE
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))
    
def n_mpjpe(predicted, target):

    assert predicted.shape == target.shape
    
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=2, keepdim=True), dim=1, keepdim=True)
    norm_target = torch.mean(torch.sum(target*predicted, dim=2, keepdim=True), dim=1, keepdim=True)
    scale = norm_target / norm_predicted
    return mpjpe(scale * predicted, target)

def weighted_bonelen_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.001 * torch.pow(predict_3d_length - gt_3d_length, 2).mean()
    return loss_length

def weighted_boneratio_loss(predict_3d_length, gt_3d_length):
    loss_length = 0.1 * torch.pow((predict_3d_length - gt_3d_length)/gt_3d_length, 2).mean()
    return loss_length

def mean_velocity_error(predicted, target):
    """
    Mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative)
    """
    assert predicted.shape == target.shape
    
    velocity_predicted = np.diff(predicted, axis=0)
    velocity_target = np.diff(target, axis=0)
    
    return np.mean(np.linalg.norm(velocity_predicted - velocity_target, axis=len(target.shape)-1))

def decoder_wloss(recon, target):

    l1_loss = F.l1_loss
    
    weight = get_w(weighted=True).cuda()
    recon_loss = l1_loss(recon, target, reduction='none')
    
    loss_wmpjpe = recon_loss * weight
    
    loss_wmpjpe = reduce(loss_wmpjpe, 'b ... -> b (...)', 'mean')
    
    return loss_wmpjpe.mean()

def decoder_tcloss(recon, target):

    weight = get_w(weighted=True).cuda()
    dif_seq = target[:,1:,:,:] - target[:,:-1,:,:]
    weights_joints = torch.ones_like(dif_seq).cuda()
    weights_mul = weight.squeeze()
    assert weights_mul.shape[0] == weights_joints.shape[-2]
    weights_joints = torch.mul(weights_joints.permute(0,1,3,2),weights_mul).permute(0,1,3,2)
    dif_seq = torch.mean(torch.multiply(weights_joints, torch.square(dif_seq)))
    loss_diff = 0.5 * dif_seq + 1.0 * mean_velocity_error_train(recon, target, axis=1)
    
    return loss_diff

def kldiv_loss(mu, logvar):
    
    kl_loss = 0.5 * torch.sum(torch.pow(mu, 2) + torch.exp(logvar) - 1.0 - logvar) * 0.000001
    
    return kl_loss

def post_processing(inputs_2d, predicted_3d, poses_3d, cam, sample=False):
    
    cam = torch.from_numpy(cam).expand((predicted_3d.shape[0],9)).cuda()
    
    uvd = torch.cat((inputs_2d, predicted_3d[:,:,:,2].unsqueeze(-1)), -1)
    xyz = get_uvd2xyz(uvd, poses_3d, cam).squeeze()
    xyz[:, 0, :] = 0
    
    poses_3d[:, :, 0] = 0
    poses_3d = rearrange(poses_3d, 'b 1 j c -> (b 1) j c')
    predicted_3d = rearrange(predicted_3d, 'b 1 j c -> (b 1) j c')
    if sample:
        predicted_3d[:,:,:2] = (0.5)*predicted_3d[:,:,:2] + (0.5)*xyz[:,:,:2]

        return rearrange(predicted_3d, '(b 1) j c -> b 1 j c')
    
    error__ = mpjpe(predicted_3d, poses_3d)
    for i in range(100):
        predicted_3d[:,:,:2] = (i/100)*predicted_3d[:,:,:2] + (1-i/100)*xyz[:,:,:2]
        result = mpjpe(predicted_3d, poses_3d)
        if i == 0:
            best_estimate = result
        else:
            if result < best_estimate:
                best_estimate = result
    
    return best_estimate
