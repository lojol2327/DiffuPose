import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument('-g', '--gpu', default='0', type=str, metavar='GPU', help='which gpu to use')
    parser.add_argument('-m', '--milestone', default='0', type=int, metavar='N', help='test')
    # General arguments
    parser.add_argument('-d', '--dataset', default='h36m', type=str, metavar='NAME', help='target dataset') # h36m or humaneva
    parser.add_argument('-k', '--keypoints', default='cpn_ft_h36m_dbb', type=str, metavar='NAME', help='2D detections to use')
    
    parser.add_argument('-str', '--subjects-train', default='S1,S5,S6,S7,S8', type=str, metavar='LIST',
                        help='training subjects separated by comma')
    parser.add_argument('-ste', '--subjects-test', default='S9,S11', type=str, metavar='LIST', help='test subjects separated by comma')
    parser.add_argument('-sun', '--subjects-unlabeled', default='', type=str, metavar='LIST',
                        help='unlabeled subjects separated by comma for self-supervision')
    parser.add_argument('-a', '--actions', default='*', type=str, metavar='LIST',
                        help='actions to train/test on, separated by comma, or * for all')
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--checkpoint-frequency', default=40, type=int, metavar='N',
                        help='create a checkpoint every N epochs')
    
    parser.add_argument('-r', '--resume', default='0', type=int, metavar='N', help='checkpoint to resume (milestone)')
    parser.add_argument('--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('--render', action='store_true', help='visualize a particular video')
    parser.add_argument('--by-subject', action='store_true', help='break down error by subject (on evaluation)')
    parser.add_argument('--export-training-curves', action='store_true', help='save training curves as .png images')
    parser.add_argument('-sptr','--subjects', default='S1,S5,S6,S7,S8,S9,S11', type=str, metavar='LIST', 
                        help='pretraining subjects separated by comma')

    # Model arguments
    parser.add_argument('-t', '--timesteps', default='100', type=int, metavar='N', help='Diffusion timesteps')
    parser.add_argument('-dim', '--model-dim', default='384', type=int, metavar='N', help='Diffusion model input embedding dimension')
    parser.add_argument('--depth', default='4', type=int, metavar='N', help='dim_mults')
    parser.add_argument('-s', '--stride', default=1, type=int, metavar='N', help='chunk size to use during training')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='batch size in terms of predicted frames')
    parser.add_argument('-drop', '--dropout', default=0., type=float, metavar='P', help='dropout probability')
    parser.add_argument('-lr', '--learning-rate', default=0.00004, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('-lrd', '--lr-decay', default=0.99, type=float, metavar='LR', help='learning rate decay per epoch')
    parser.add_argument('-no-da', '--no-data-augmentation', dest='data_augmentation', action='store_false',
                        help='disable train-time flipping')
    parser.add_argument('-frame', '--number-of-frames', default='1', type=int, metavar='N',
                        help='how many frames used as input')
    parser.add_argument('-joint', '--number-of-joints', default='17', type=int, metavar='N',
                        help='how many joints exist in data')

    # Experimental
    parser.add_argument('--test-load', type=str, metavar='PATH', help='model to test')
    parser.add_argument('--num-sample', default=1, type=int, metavar='N', help='number of diffusion sample when evaluation')
    parser.add_argument('--subset', default=1, type=float, metavar='FRACTION', help='reduce dataset size by fraction')
    parser.add_argument('--downsample', default=1, type=int, metavar='FACTOR', help='downsample frame rate by factor (semi-supervised)')
    
    # Visualization
    parser.add_argument('--viz-subject', type=str, metavar='STR', help='subject to render')
    parser.add_argument('--viz-action', type=str, metavar='STR', help='action to render')
    parser.add_argument('--viz-camera', type=int, default=0, metavar='N', help='camera to render')
    parser.add_argument('--viz-video', type=str, metavar='PATH', help='path to input video')
    parser.add_argument('--viz-skip', type=int, default=0, metavar='N', help='skip first N frames of input video')
    parser.add_argument('--viz-output', type=str, metavar='PATH', help='output file name (.gif or .mp4)')
    parser.add_argument('--viz-export', type=str, metavar='PATH', help='output file name for coordinates')
    parser.add_argument('--viz-bitrate', type=int, default=3000, metavar='N', help='bitrate for mp4 videos')
    parser.add_argument('--viz-no-ground-truth', action='store_true', help='do not show ground-truth poses')
    parser.add_argument('--viz-limit', type=int, default=-1, metavar='N', help='only render first N frames')
    parser.add_argument('--viz-downsample', type=int, default=1, metavar='N', help='downsample FPS by a factor N')
    parser.add_argument('--viz-size', type=int, default=5, metavar='N', help='image size')
    
    parser.set_defaults(bone_length_term=True)
    parser.set_defaults(data_augmentation=True)
    parser.set_defaults(test_time_augmentation=True)
    # parser.set_defaults(test_time_augmentation=False)

    args = parser.parse_args()
    # Check invalid configuration
    if args.resume and args.evaluate:
        print('Invalid flags: --resume and --evaluate cannot be set at the same time')
        exit()
        
    if args.export_training_curves and args.no_eval:
        print('Invalid flags: --export-training-curves and --no-eval cannot be set at the same time')
        exit()

    return args
