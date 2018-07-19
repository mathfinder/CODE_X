import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import torch

def str2bool(v):
    return v.lower() in ('true')

def main(config):

    if len(config.device_ids) > 0:
        torch.cuda.set_device(config.device_ids[0])

    # For fast training
    cudnn.benchmark = True

    # Create directories if not exist
    if not os.path.exists(config.ckpt_path):
        os.makedirs(config.ckpt_path)
    if not os.path.exists(os.path.join(config.ckpt_path, config.task_name, config.log_path)):
        os.makedirs(os.path.join(config.ckpt_path, config.task_name, config.log_path))
    if not os.path.exists(os.path.join(config.ckpt_path, config.task_name, config.model_save_path)):
        os.makedirs(os.path.join(config.ckpt_path, config.task_name, config.model_save_path))
    if not os.path.exists(os.path.join(config.ckpt_path, config.task_name, config.sample_path)):
        os.makedirs(os.path.join(config.ckpt_path, config.task_name, config.sample_path))
    if not os.path.exists(os.path.join(config.ckpt_path, config.task_name, config.result_path)):
        os.makedirs(os.path.join(config.ckpt_path, config.task_name, config.result_path))
    if not os.path.exists(os.path.join(config.ckpt_path, config.task_name, config.evaluates)):
        os.makedirs(os.path.join(config.ckpt_path, config.task_name, config.evaluates))

    # Data loader
    loader1 = None
    loader2 = None
    eval_loader = None

    if config.dataset1 =='ImageNet':
        loader1 = get_loader(config.dataset_image_path1,config.dataset_crop_size,
                                   config.image_size, config.batch_size, config.dataset1, config.mode, config.num_workers)
        eval_loader = get_loader(config.dataset_image_path1,config.dataset_crop_size,
                                   config.image_size, config.batch_size, config.dataset1, 'test', config.num_workers)
    if config.dataset2 =='AgePairDataFolder':
        loader2 = get_loader(config.dataset_image_path2,config.dataset_crop_size,
                                   config.image_size, config.batch_size, config.dataset2, config.mode, config.num_workers)


    # Solver
    solver = Solver(loader1, loader2, eval_loader, config)

    if config.mode == 'train':
        solver.train()

    elif config.mode == 'test':
        solver.test()

    elif config.mode == 'evaluate':
        solver.evaluate()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #
    parser.add_argument('--device_ids', default=[0])

    # Model hyper-parameters
    parser.add_argument('--c_dim', type=int, default=7)
    parser.add_argument('--c2_dim', type=int, default=7)
    parser.add_argument('--dataset_crop_size', type=int, default=128)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--g_conv_dim', type=int, default=64)
    parser.add_argument('--d_conv_dim', type=int, default=64)
    parser.add_argument('--g_repeat_num', type=int, default=6)
    parser.add_argument('--d_repeat_num', type=int, default=6)
    parser.add_argument('--d2_repeat_num', type=int, default=3)
    parser.add_argument('--n_hiddens', type=int, default=8)

    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0001)
    parser.add_argument('--lambda_cls', type=float, default=1)
    parser.add_argument('--lambda_rec', type=float, default=10)
    parser.add_argument('--lambda_reg', type=float, default=10)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--lambda_kl', type=float, default=1)
    parser.add_argument('--debug', type=bool, default=True)

    parser.add_argument('--d_train_repeat', type=int, default=5)

    # Training settings
    parser.add_argument('--dataset1', type=str, default='ImageNet', choices=['ImageNet'])
    parser.add_argument('--dataset2', type=str, default='AgePairDataFolder', choices=['AgePairDataFolder', None])
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_epochs_decay', type=int, default=0)
    parser.add_argument('--num_iters', type=int, default=200000)
    parser.add_argument('--num_iters_decay', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--pretrained_model', type=str, default='')
    parser.add_argument('--resume', type=str, default='')

    # Evaluation settings
    parser.add_argument('--eval_batchsize', type=int, default=20,
                        help='Batch size to use')
    parser.add_argument('--dims', type=int, default=512,
                        help=('Dimensionality of Inception features to use. '
                              'By default, uses pool3 features'))
    parser.add_argument('--eval_model', type=str, default='resnet34',
                        help=('model for calculate FD'))

    # Test settings
    parser.add_argument('--test_model', type=str, default='200000')
    parser.add_argument('--test_interval', type=list, default=[1, 1])
    parser.add_argument('--test_iters', type=str, default='8300')
    parser.add_argument('--test_step', type=int, default=20)
    parser.add_argument('--test_traverse', type=bool, default=True)

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'demo', 'evaluate', 'vis'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--with_mask', type=bool, default=False)

    # Path
    # parser.add_argument('--task_name', type=str,
    #                     default='c_gan_v2__reg_D__weighted_multi_field_G__no_featurewise_pair_D__lambda_rec=10_lambda_reg=1__multi_sample')
    parser.add_argument('--task_name', type=str, default='cos_big_gauss=1')

    parser.add_argument('--dataset_image_path1', type=str, default='./data/faceAgingACMMM')
    parser.add_argument('--dataset_image_path2', type=str, default='./data/ItelyFaceAging')
    parser.add_argument('--metadata_path', type=str, default='./data/list_attr_celeba.txt')
    parser.add_argument('--test_label_path', type=str, default='/home/ben/mathfinder/PROJECT/IJCV2018/code/StarGAN-Multi/data/CelebA_nocrop/test_label_list_1.txt')
    parser.add_argument('--ckpt_path', type=str, default='ckpt')
    parser.add_argument('--log_path', type=str, default='logs')
    parser.add_argument('--model_save_path', type=str, default='models')
    parser.add_argument('--sample_path', type=str, default='samples')
    parser.add_argument('--result_path', type=str, default='results')
    parser.add_argument('--evaluates', type=str, default='evaluates')

    # Step size
    parser.add_argument('--log_step', type=int, default=500)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10)
    parser.add_argument('--model_save_star', type=int, default=0)

    config = parser.parse_args()

    print(config)
    main(config)
