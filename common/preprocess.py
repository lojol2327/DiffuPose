import numpy as np


import os
import errno
from pathlib import Path

from common.args import parse_args
from common.camera import *
from common.utils import *

args = parse_args()
# print(args)

    
try:
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)
    
print('Loading Dataset..')

''' One can read [loaded npz file] by calling [.files] method. '''


dataset_path = Path('./data/data_3d_' + args.dataset + '.npz')
assert os.path.exists(dataset_path)

if args.dataset == 'h36m':
    if args.keypoints == 'hr':
        from dataset.h36m_dataset_hr import Human36mDataset
    else:
        from dataset.h36m_dataset import Human36mDataset
    dataset = Human36mDataset(dataset_path)
elif args.dataset.startswith('humaneva'):
    from dataset.humaneva_dataset import HumanEvaDataset
    dataset = HumanEvaDataset(dataset_path)
elif args.dataset.startswith('custom'):
    from dataset.custom_dataset import CustomDataset
    dataset = CustomDataset('.data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz')
else:
    raise KeyError('Invalid dataset')

print('Preparing data..')

'''Append camera coordinate'''
for subject in dataset.subjects():  
    for action in dataset[subject].keys():  
        anim = dataset[subject][action] 
        
        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1] #Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            # add anim 'position_3d', which contains 3D positions represented in camera coordinates
            anim['positions_3d'] = positions_3d
            
''' Successfully added 'position_3d(camera coord.) to 'dataset'. '''

print('Loading 2D detections...')
keypoints = np.load('./data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz', allow_pickle=True) # Load data!
if 'metadata' in keypoints.keys():
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    if 'layout_name' not in keypoints_metadata:
        keypoints_metadata['layout_name']='none'
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()
else:
    if args.keypoints == 'hr':
        keypoints_symmetry = [[4, 5, 6, 10, 11, 12], [1, 2, 3, 13, 14, 15]]
        kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
        keypoints = keypoints['positions_2d'].item()
        keypoints_metadata = {'layout_name': 'h36m', 'keypoints_symmetry': keypoints_symmetry, 'num_joints': 16}
        
for subject in dataset.subjects():
    assert subject in keypoints, f'Subject {subject} is missing from the 2D detections dataset.'
    for action in dataset[subject].keys():
        assert action in keypoints[subject], f'Action {action} of subject {subject} is missing from the 2D detections dataset.'
        if 'positions_3d' not in dataset[subject][action]: 
            continue
        
        # 4 Cameras exists per action
        for cam_idx in range(len(keypoints[subject][action])):
            
            # We check for >= instead of == because some videos in H3.6M contain extra frames
            # dataset[subject][action] -> {'positions' : array ..., 'cameras' : array ..., 'positions_3d' : array ...}  (dict)
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length
            
            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]
                
        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])
        
'''Now we obtained '3D camera coordinates' and '2D detected coordinates' of matched length.'''

for subject in keypoints.keys():
    for action in keypoints[subject]:
        # 4 Cameras
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

# subjects = args.subjects_pretrain.split(',') # 'Exploit ALL subsets'    
subjects_train = args.subjects_train.split(',') # 'S1,S5,S6,S7,S8'
subjects_semi = [] if not args.subjects_unlabeled else args.subjects_unlabeled.split(',') # default = ''

'args.render -> default : False'

if not args.render:
    subjects_test = args.subjects_test.split(',') # 'S9, S11'
else:
    subjects_test = [args.viz_subject]
    

def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    '''
    fetch

    returns camera intrinsic parameters(length 9), 2d, 3d (list form)

    2D & 3D joints w.r.t. camera coordinates for ALL subjects are already prepared.
    fecth fn. extracts 2D,3D,camera info. of designated dataset.(e.g., train, test, ..)
    + Action filtering
    '''
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue
            
            'Extract 2D coordinates w.r.t 4 cameras.'
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)): # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)): # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])
                    
    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None
        
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]
                
    return out_camera_params, out_poses_3d, out_poses_2d

action_filter = None if args.actions == '*' else args.actions.split(',')
if action_filter is not None:
    print('Selected actions:', action_filter)
    
# subjects_test : S9, S11
cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, action_filter)

###########

receptive_field = args.number_of_frames
print(f'[INFO] Receptive field: {receptive_field} frames.')
# pad = (receptive_field -1) // 2
pad = 0
min_loss = 300000
width_res = cam['res_w']
height_res = cam['res_h']
if args.dataset == 'h36m':
    n_joints = 17
else:
    n_joints = keypoints_metadata['num_joints']

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    '''
    inputs_2d & inputs_3d : [1, sequence_length, j, c]
    for seq2seq inference, we should prepare data
    with the shape of [B, F, J, C]
    '''
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = torch.squeeze(inputs_3d)
    
    assert inputs_2d.shape[0] == inputs_3d.shape[0]
    
    remainder = inputs_2d_p.shape[0] % receptive_field  # 2962 % 64 = 18
    if remainder == 0:
        groups = inputs_2d_p.shape[0] // receptive_field
    else:
        groups = (inputs_2d_p.shape[0] // receptive_field) + 1 # 2962 // 64 = 46

    eval_input_2d = torch.empty(groups, receptive_field, inputs_2d.shape[-2], inputs_2d.shape[-1])  # (47, 64, 17, 2)
    poses_3d = torch.empty(groups, receptive_field, inputs_3d.shape[-2], inputs_3d.shape[-1]) # (47, 64, 17, 3)
    
    for i in range(groups):
        if remainder == 0:
            eval_input_2d[i, :, :, :] = inputs_2d_p[i*receptive_field : (i+1)*receptive_field, :, :]
            poses_3d[i, :, :, :] = inputs_3d_p[i*receptive_field : (i+1)*receptive_field, :, :]
        else:
            if i < groups -1:
                eval_input_2d[i, :, :, :] = inputs_2d_p[i*receptive_field : (i+1)*receptive_field, :, :]
                poses_3d[i, :, :, :] = inputs_3d_p[i*receptive_field : (i+1)*receptive_field, :, :]
            else:
                eval_input_2d[i, :, :, :] = inputs_2d_p[-receptive_field:, :, :]
                poses_3d[i, :, :, :] = inputs_3d_p[-receptive_field, :, :]
    
    assert poses_3d.shape == (groups, receptive_field, inputs_3d.shape[-2], inputs_3d.shape[-1]) \
            and eval_input_2d.shape == (groups, receptive_field, inputs_2d.shape[-2], inputs_2d.shape[-1]), 'Shape does not match as intended.'

    return eval_input_2d, poses_3d
