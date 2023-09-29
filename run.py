import os

from model.diffusion import Diffusion
from model.modulated_gcn import ModulatedGCN as MGCN

from common.base import Trainer, Tester
from common.args import parse_args
from common.preprocess import dataset

from torch.nn.parallel.data_parallel import DataParallel

from thop import profile


from graph_utils import *

args = parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
results_folder = './results'
viz_folder = './viz'

adj = adj_mx_from_skeleton(dataset.skeleton())
denoise_fn = MGCN(adj, args.model_dim, num_layers=4, p_dropout=0., nodes_group=None).cuda()

if args.keypoints == 'hr':
    n_joint = 16
else:
    n_joint = args.number_of_joints

diffusion = Diffusion(
    denoise_fn,
    n_frame = 1,
    n_joint = n_joint,
    timesteps = args.timesteps,
    loss_type = 'l1'
).cuda()

diffusion = DataParallel(diffusion).cuda()

#######################################
# input = torch.randn(1, args.number_of_frames, args.number_of_joints, 3).cuda()
# cond = torch.randn(1, args.number_of_frames, args.number_of_joints, 2).cuda()
# macs, _ = profile(diffusion.cuda(), inputs=(input, cond))
# print('{:<30} {:<8} {:<15}'.format('Computational complexity: ', macs/1000000*2/args.number_of_frames, 'Million FLOPs per frame'))
#######################################

if not args.test_load:
    print('Training...')
    trainer = Trainer(
        diffusion,
        results_folder,
        train_batch_size = args.number_of_frames * args.batch_size,
        train_lr = args.learning_rate,
        train_num_steps = 310000,
        gradient_accumulate_every = 2,    
        ema_decay = 0.995,                
        amp = True,                   
    )
    model_params = 0
    for parameter in diffusion.parameters():
        model_params += parameter.numel()
    print(f'INFO: Trainable parameter count: {model_params}')
    print(f'INFO: Using GPU {args.gpu}')

    trainer.train()
else:
    tester = Tester(
        diffusion,
        args.test_load,
        results_folder,
        viz_folder,
        ema_decay = 0.995,
    )
    
    print(f'INFO: Using GPU {args.gpu}')
    if args.render:
        print('Rendering...')
        tester.render()
    else:
        print('Evaluating start...')
        tester.test()