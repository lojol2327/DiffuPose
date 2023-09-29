import numpy as np
import torch

from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from ema_pytorch import EMA

from tqdm import tqdm
from pathlib import Path


from common.preprocess import *
from common.generator import ChunkedGenerator, UnchunkedGenerator
from common.args import parse_args
from common.loss import *

from einops import rearrange

args = parse_args()

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        results_folder,
        *,
        ema_decay = 0.995, 
        train_batch_size = 32,
        train_lr = 4e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 2,
        amp = False,
        step_start_ema = 2000,
        ema_update_every = 10,
        save_and_sample_every = 1000,
        augment_horizontal_flip = True
    ):
        super().__init__()
        
        self.model = diffusion_model

        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.opt = Adam(diffusion_model.parameters(), lr = train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled = amp)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        if args.resume:
            data = torch.load(str(self.results_folder / f'model-{args.resume}.pt'))
            self.step = data['step']
            self.model.load_state_dict(data['model'])
            self.ema.load_state_dict(data['ema'])         

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.scaler.state_dict()
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))
    
    def _make_train_generator(self):
        cameras_train, poses_train, poses_train_2d = fetch(subjects_train, action_filter, subset=args.subset)
        causal_shift = 0
        train_generator = ChunkedGenerator(self.batch_size//args.number_of_frames, cameras_train, poses_train, poses_train_2d, args.number_of_frames,
                                       pad=pad, causal_shift=causal_shift, shuffle=True, augment=args.data_augmentation,
                                       kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        
        return train_generator.next_epoch()
    
    def train(self):
        
        with tqdm(initial = self.step, total = self.train_num_steps) as pbar:
            train_generator = self._make_train_generator()
            
            while self.step < self.train_num_steps:
                for _ in range(self.gradient_accumulate_every):
                    try:
                        cameras_train, batch_3d, batch_2d = next(train_generator)
                    except StopIteration:
                        train_generator = self._make_train_generator()
                        cameras_train, batch_3d, batch_2d = next(train_generator)
                    cameras_train = torch.from_numpy(cameras_train.astype('float32'))
                    poses_3d = torch.from_numpy(batch_3d.astype('float32'))
                    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                    
                    if torch.cuda.is_available():
                        poses_3d = poses_3d.cuda()
                        inputs_2d = inputs_2d.cuda()
                    poses_traj = poses_3d[:, :, :1].clone()
                    poses_3d[:, :, 0] = 0
                    
                    with autocast(enabled = self.amp):
                        w_loss, loss = self.model(poses_3d, inputs_2d)
                        self.scaler.scale(w_loss / self.gradient_accumulate_every).mean().backward()

                    pbar.set_description(f'w_loss, loss: {w_loss.mean().item():.4f} {loss.mean().item():.4f}')
                                    
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()

                self.ema.update()

                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                self.step += 1
                pbar.update(1)

        print('training complete')
        
class Tester(object):
    def __init__(
        self,
        diffusion_model,
        test_model,
        results_folder,
        video_folder,
        ema_decay = 0.995,
        ema_update_every = 10,
    ):
        super().__init__()
    
        all_actions = {}
        all_actions_by_subject = {}
        for subject in subjects_test:
            if subject not in all_actions_by_subject:
                all_actions_by_subject[subject] = {}
                
            for action in dataset[subject].keys():
                action_name = action.split(' ')[0]
                if action_name not in all_actions:
                    all_actions[action_name] = []
                if action_name not in all_actions_by_subject[subject]:
                    all_actions_by_subject[subject][action_name] = []
                all_actions[action_name].append((subject, action))
                all_actions_by_subject[subject][action_name].append((subject, action))
        
        self.all_actions = all_actions
        self.all_actions_by_subject = all_actions_by_subject
        
        self.model = diffusion_model
        self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)
        
        if video_folder is not None:
            self.video_folder = Path(video_folder)
            self.video_folder.mkdir(exist_ok = True)
        
        data = torch.load(str(self.results_folder / args.test_load))
        print(f'Testing with best model...')

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema.load_state_dict(data['ema'])
            
    
    def fetch_actions(self, actions):
        out_poses_3d = []
        out_poses_2d = []
        out_camera_params = []
        
        for subject, action in actions:
            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])
                
            poses_3d = dataset[subject][action]['positions_3d']
            assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
            for i in range(len(poses_3d)):
                out_poses_3d.append(poses_3d[i])
                
            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])


        stride = args.downsample # default : 1
        if stride > 1:
            # Downsample as requested
            for i in range(len(out_poses_2d)):
                out_poses_2d[i] = out_poses_2d[i][::stride]
                if out_poses_3d is not None:
                    out_poses_3d[i] = out_poses_3d[i][::stride]
                    
        return out_camera_params, out_poses_3d, out_poses_2d

    def _make_test_generator(self, actions, action_key):
        cam, poses_act, poses_2d_act = self.fetch_actions(actions[action_key])
        causal_shift = 0
        test_generator = UnchunkedGenerator(cam, poses_act, poses_2d_act, pad=pad, causal_shift=causal_shift, augment=args.test_time_augmentation,
                                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
            
        return test_generator.next_epoch()
    
    def evaluate(self, test_generator, action=None, return_predictions=False, use_trajectory_model=False):
        epoch_loss_3d_pos = 0
        epoch_loss_3d_pos_procrustes = 0
        epoch_loss_3d_pos_scale = 0
        epoch_loss_3d_vel = 0
        with torch.no_grad():
            if not use_trajectory_model:
                self.ema.ema_model.eval()
            
            N = 0
            for cam, batch_3d, batch_2d in test_generator:
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                inputs_3d = torch.from_numpy(batch_3d.astype('float32'))
                
                # seq_len = inputs_3d.shape[1] # inputs_3d : [ 1, seq_len, 17, 3 ]
                # remainder = seq_len % receptive_field
                
                ##### apply test-time-augmentation (following Videopose3d)
                inputs_2d_flip = inputs_2d.clone()
                inputs_2d_flip[:, :, :, 0] *= -1
                inputs_2d_flip[:, :, kps_left + kps_right, :] = inputs_2d_flip[:, :, kps_right + kps_left, :]
                
                ##### convert size
                inputs_2d, poses_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
                inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
                
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()
                    inputs_2d_flip = inputs_2d_flip.cuda()
                    poses_3d = poses_3d.cuda()
                
                poses_3d_ = poses_3d.clone()
                poses_3d[:, :, 0] = 0
                
                poses = []
                for _ in range(args.num_sample):
                    predicted_3d_pos = self.ema.ema_model.module.sample(cond=inputs_2d, batch_size=poses_3d.shape[0])
                    predicted_3d_pos_flip = self.ema.ema_model.module.sample(cond=inputs_2d_flip, batch_size=poses_3d.shape[0])
                    predicted_3d_pos_flip[:, :, :, 0] *= -1
                    predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]
                    poses.append(predicted_3d_pos)
                    poses.append(predicted_3d_pos_flip)

                predicted_3d_pos = torch.stack(poses, dim=1)
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=1)
                
                torch.cuda.empty_cache()
                
                if return_predictions:
                    return predicted_3d_pos.cpu().numpy()
                
                ' predicted_3d_pos & poses_3d : [B x J x C] '
                
                error = post_processing(inputs_2d, predicted_3d_pos, poses_3d_, cam)

                epoch_loss_3d_pos_scale += poses_3d.shape[0]*poses_3d.shape[1] * n_mpjpe(predicted_3d_pos, poses_3d).item()
                
                epoch_loss_3d_pos += poses_3d.shape[0]*poses_3d.shape[1] * error.item()
                N += poses_3d.shape[0] * poses_3d.shape[1]

                inputs = poses_3d.cpu().numpy().reshape(-1, poses_3d.shape[-2], poses_3d.shape[-1])
                predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, poses_3d.shape[-2], poses_3d.shape[-1])

                epoch_loss_3d_pos_procrustes += poses_3d.shape[0]*poses_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

                # Compute velocity error
                epoch_loss_3d_vel += poses_3d.shape[0]*poses_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)
                
        if action is None:
            print('----------')
        else:
            print('----'+action+'----')
        e1 = (epoch_loss_3d_pos / N)*1000
        e2 = (epoch_loss_3d_pos_procrustes / N)*1000
        e3 = (epoch_loss_3d_pos_scale / N)*1000
        ev = (epoch_loss_3d_vel / N)*1000
        print('Protocol #1 Error (MPJPE):', e1, 'mm')
        print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
        print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
        print('Velocity Error (MPJVE):', ev, 'mm')
        print('----------')

        
        return e1, e2, e3, ev    

    def render(self):
        input_keypoints = keypoints[args.viz_subject][args.viz_action][args.viz_camera].copy()
        ground_truth = None
        if args.viz_subject in dataset.subjects() and args.viz_action in dataset[args.viz_subject]:
            if 'positions_3d' in dataset[args.viz_subject][args.viz_action]:
                ground_truth = dataset[args.viz_subject][args.viz_action]['positions_3d'][args.viz_camera].copy()
        if ground_truth is None:
            print('INFO: this action is unlabeled. Ground truth will not be rendered.')
        
        gen = UnchunkedGenerator(None, [ground_truth], [input_keypoints],
                                pad=pad, causal_shift=0, augment=args.test_time_augmentation,
                                kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
        prediction = self.evaluate(gen.next_epoch(), return_predictions=True)

        if args.viz_export is not None:
            print('Exporting joint positions to', args.viz_export)
            # Predictions are in camera space
            np.save(args.viz_export, prediction)
            
        if args.viz_output is not None:
            if ground_truth is not None:
                # Reapply trajectory
                prediction = prediction.squeeze()
                trajectory = ground_truth[:, :1]
                ground_truth[:, 1:] += trajectory
                prediction += trajectory
                
            # Invert camera transformation
            cam = dataset.cameras()[args.viz_subject][args.viz_camera]
            if ground_truth is not None:
                prediction = camera_to_world(prediction, R=cam['orientation'], t=cam['translation'])
                ground_truth = camera_to_world(ground_truth, R=cam['orientation'], t=cam['translation'])
            else:
                # If the ground truth is not available, take the camera extrinsic params from a random subject.
                # They are almost the same, and anyway, we only need this for visualization purposes.
                for subject in dataset.cameras():
                    if 'orientation' in dataset.cameras()[subject][args.viz_camera]:
                        rot = dataset.cameras()[subject][args.viz_camera]['orientation']
                        break
                prediction = camera_to_world(prediction, R=rot, t=0)
                # We don't have the trajectory, but at least we can rebase the height
                prediction[:, :, 2] -= np.min(prediction[:, :, 2])
            
            anim_output = {'Reconstruction': prediction}
            if ground_truth is not None and not args.viz_no_ground_truth:
                anim_output['Ground truth'] = ground_truth
                
            input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])
            
            from common.visualization import render_animation

            render_animation(input_keypoints, keypoints_metadata, anim_output,
                            dataset.skeleton(), dataset.fps(), args.viz_bitrate, cam['azimuth'], args.viz_output,
                            limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                            input_video_path=args.viz_video, viewport=(cam['res_w'], cam['res_h']),
                            input_video_skip=args.viz_skip)
                
    def run_evaluation(self, actions, action_filter=None):
        errors_p1 = []
        errors_p2 = []
        errors_p3 = []
        errors_vel = []
        
        for action_key in actions.keys():
            if action_filter is not None:
                found = False 
                for a in action_filter:
                    if action_key.startswith(a):
                        found = True
                        break
                if not found:
                    continue
                
            test_generator = self._make_test_generator(actions, action_key)
            e1, e2, e3, ev = self.evaluate(test_generator, action_key)
            errors_p1.append(e1)
            errors_p2.append(e2)
            errors_p3.append(e3)
            errors_vel.append(ev)
        
        print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
        print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
        print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
        print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')

        
    def test(self):
        if not args.by_subject:
            self.run_evaluation(self.all_actions, action_filter)
        else:
            for subject in self.all_actions_by_subject.keys():
                print('Evaluating on subject', subject)
                self.run_evaluation(self.all_actions_by_subject[subject], action_filter)
                print('')
                
    
