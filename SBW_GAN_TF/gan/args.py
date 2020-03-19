import argparse
from argparse import Namespace


def parser_with_default_args():
    """
        Define args that is used in default wgan_gp, you can add other args in client.
    """
    parser = argparse.ArgumentParser(description="Improved Wasserstein GAN implementation for Keras.")
    parser.add_argument("--output_dir", default='output/generated_samples', help="Directory with generated sample images")
    parser.add_argument("--batch_size", default=64, type=int, help='Size of the batch')
    parser.add_argument("--training_ratio", default=5, type=int,
                        help="The training ratio is the number of discriminator updates per generator update." +
                        "The paper uses 5")
    parser.add_argument("--gradient_penalty_weight", default=0, type=float, help='Weight of gradient penalty loss')

    parser.add_argument("--generator_adversarial_objective", default='hinge', choices=['ns-gan', 'lsgan', 'wgan', 'hinge'])
    parser.add_argument("--discriminator_adversarial_objective", default='hinge', choices=['ns-gan', 'lsgan', 'wgan', 'hinge'])
    parser.add_argument("--gradient_penalty_type", default='wgan-gp', choices=['wgan-gp', 'dragan'])

    parser.add_argument("--number_of_epochs", default=50, type=int, help="Number of training epochs")
    
    parser.add_argument("--checkpoints_dir", default="output/checkpoints", help="Folder with checkpoints")
    parser.add_argument("--checkpoint_ratio", default=10, type=int, help="Number of epochs between consecutive checkpoints")    
    parser.add_argument("--generator_checkpoint", default=None, help="Previosly saved model of generator")
    parser.add_argument("--discriminator_checkpoint", default=None, help="Previosly saved model of discriminator")
    
    parser.add_argument("--display_ratio", default=1, type=int,  help='Number of epochs between ploting')
    parser.add_argument("--start_epoch", default=0, type=int, help='Start epoch for starting from checkpoint')

    parser.add_argument("--name", default="gan", help="Name of the experiment (it will create corresponding folder)")
    parser.add_argument("--phase", choices=['train', 'test'], default='train',
                        help="Train or test, test only compute scores and generate grid of images."
                             "For test generator checkpoint should be given.")

    parser.add_argument("--dataset", default='cifar10',
                        choices=['mnist', 'cifar10', 'cifar100', 'fashion-mnist', 'stl10', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to train on')
    parser.add_argument("--arch", default='dcgan', choices=['res', 'dcgan'], help="Gan architecture resnet or dcgan.")

    parser.add_argument("--generator_lr", default=2e-4, type=float, help="Learning rate")
    parser.add_argument("--discriminator_lr", default=2e-4, type=float, help="Learning rate")

    parser.add_argument("--beta1", default=0, type=float, help='Adam parameter')
    parser.add_argument("--beta2", default=0.9, type=float, help='Adam parameter')
    parser.add_argument("--lr_decay_schedule", default='linear',
                        help='Learnign rate decay schedule:'
                             'None - no decay.'
                             'linear - linear decay to zero.'
                             'half-linear - linear decay to 0.5'
                             'linear-end - constant until 0.9, then linear decay to 0. '
                             'dropat30 - drop lr 10 times at 30 epoch (any number insdead of 30 allowed).')

    parser.add_argument("--generator_spectral", default=0, type=int, help='Use spectral norm in generator.')
    parser.add_argument("--discriminator_spectral", default=0, type=int, help='Use spectral norm in discriminator.')

    parser.add_argument("--fully_diff_spectral", default=0, type=int,
                        help='Fully difirentiable spectral normalization.')
    parser.add_argument("--spectral_iterations", default=1, type=int, help='Number of iteration per spectral update.')
    parser.add_argument("--conv_singular", default=0, type=int, help='Use convolutional spectral normalization.')

    parser.add_argument("--gan_type", default=None, choices=[None, 'AC_GAN', 'PROJECTIVE'],
                        help='Type of gan to use. None for unsuperwised.')

    parser.add_argument("--filters_emb", default=10, type=int, help='Number of inner filters in factorized conv.')

    parser.add_argument("--generator_block_norm", default='d', choices=['n', 'b', 'd', 'dr'],
                        help='Normalization in generator block. b - batch, d - whitening, n - none, '
                             'dr - whitening with renornaliazation.')
    parser.add_argument("--generator_block_coloring", default='uconv',
                        choices=['ccs', 'fconv', 'ucs', 'uccs', 'ufconv', 'cconv', 'uconv', 'ucconv', 'ccsuconv', 'n'],
                        help="Layer after block normalization. ccs - conditional shift and scale."
                             "ucs - uncoditional shift and scale. ucconv - condcoloring. ufconv - condcoloring + sa."
                             "n - None.")
    parser.add_argument("--generator_last_norm", default='d', choices=['n', 'b', 'd', 'dr'],
                        help='Normalization in generator block. b - batch, d - whitening, n - none, '
                             'dr - whitening with renornaliazation.')
    parser.add_argument("--generator_last_coloring", default='ucconv',
                        choices=['ccs', 'ucs', 'uccs', 'ufconv', 'cconv', 'uconv', 'ucconv', 'ccsuconv', 'n'],
                        help="Layer after block normalization. ccs - conditional shift and scale."
                             "ucs - uncoditional shift and scale. ucconv - condcoloring. ufconv - condcoloring + sa."
                             "n - None.")
    parser.add_argument("--d_instance_norm", default=0, type=int, choices=[0, 1], help='0:false 1:true')
    parser.add_argument("--d_decomposition", default='cholesky', choices=['cholesky', 'zca', 'pca', 'iter_norm','cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm'], help='')
    parser.add_argument("--d_whitten_m", default=0, type=int, help='')
    parser.add_argument("--d_coloring_m", default=0, type=int, help='')
    parser.add_argument("--d_iter_num", default=5, type=int, help='')
    parser.add_argument("--d_before_conv", default=0, type=int, help='')

    parser.add_argument("--g_instance_norm", default=0, type=int, choices=[0, 1], help='0:false 1:true')
    parser.add_argument("--g_decomposition", default='cholesky', choices=['cholesky', 'zca', 'pca', 'iter_norm','cholesky_wm', 'zca_wm', 'pca_wm', 'iter_norm_wm'],
                        help='')
    parser.add_argument("--g_whitten_m", default=0, type=int, help='')
    parser.add_argument("--g_coloring_m", default=0, type=int, help='')
    parser.add_argument("--g_iter_num", default=5, type=int, help='')
    parser.add_argument("--g_before_conv", default=0, type=int, help='')
    parser.add_argument("--generator_batch_multiple", default=2, type=int,
                        help="Size of the generator batch, multiple of batch_size.")
    parser.add_argument("--generator_concat_cls", default=0, type=int, help='Concat labels to noise in generator.')
    parser.add_argument("--generator_filters", default=256, type=int, help='Base number of filters in generator block.')

    parser.add_argument("--discriminator_norm", default='n', choices=['n', 'b', 'd', 'dr'],
                        help='Normalization in disciminator block. b - batch, d - whitening, n - none, '
                             'dr - whitening with renornaliazation.')
    parser.add_argument("--discriminator_coloring", default='n',
                        choices=['ccs', 'fconv', 'ucs', 'uccs', 'ufconv', 'cconv', 'uconv', 'ucconv', 'ccsuconv', 'n'],
                        help="Layer after block normalization. ccs - conditional shift and scale."
                             "ucs - uncoditional shift and scale. ucconv - condcoloring. ufconv - condcoloring + sa."
                             "n - None.")
    parser.add_argument("--discriminator_filters", default=128, type=int,
                        help='Base number of filters in discriminator block.')
    parser.add_argument("--discriminator_dropout", type=float, default=0, help="Use dropout in discriminator.")
    parser.add_argument("--shred_disc_batch", type=int, default=0, help='Shred batch in discriminator to save memory')

    parser.add_argument("--sum_pool", default=1, type=int, help='Use sum or average pooling in discriminator.')

    parser.add_argument("--samples_inception", default=50000, type=int,
                        help='Samples for IS score, 0 - no compute inception')
    parser.add_argument("--samples_fid", default=10000, type=int, help="Samples for FID score, 0 - no compute FID")

    args = parser.parse_args()

    return args


def get_generator_params(args):
    params = Namespace()
    params.output_channels = 1 if args.dataset.endswith('mnist') else 3
    params.input_cls_shape = (1,)

    first_block_w = (7 if args.dataset.endswith('mnist') else (6 if args.dataset == 'stl10' else 4))
    params.first_block_shape = (first_block_w, first_block_w, args.generator_filters)
    if args.arch == 'res':
        if args.dataset == 'tiny-imagenet':
            params.block_sizes = [args.generator_filters, args.generator_filters, args.generator_filters,
                                  args.generator_filters]
            params.resamples = ("UP", "UP", "UP", "UP")
        elif args.dataset.endswith('imagenet'):
            params.block_sizes = [args.generator_filters, args.generator_filters,
                                  args.generator_filters, args.generator_filters / 2, args.generator_filters / 4]

            params.resamples = ("UP", "UP", "UP", "UP", "UP")
        else:
            params.block_sizes = tuple([args.generator_filters] * 2) if args.dataset.endswith('mnist') else tuple(
                [args.generator_filters] * 3)
            params.resamples = ("UP", "UP") if args.dataset.endswith('mnist') else ("UP", "UP", "UP")
    else:
        assert args.dataset != 'imagenet'
        params.block_sizes = ([args.generator_filters, args.generator_filters / 2] if args.dataset.endswith('mnist')
                              else [args.generator_filters, args.generator_filters / 2, args.generator_filters / 4])
        params.resamples = ("UP", "UP") if args.dataset.endswith('mnist') else ("UP", "UP", "UP")
    params.number_of_classes = 100 if args.dataset == 'cifar100' else (1000 if args.dataset == 'imagenet'
                                                                       else (
        200 if args.dataset == 'tiny-imagenet' else 10))

    params.concat_cls = args.generator_concat_cls

    params.block_norm = args.generator_block_norm
    params.block_coloring = args.generator_block_coloring

    params.last_norm = args.generator_last_norm
    params.last_coloring = args.generator_last_coloring

    params.decomposition = args.g_decomposition
    params.whitten_m = args.g_whitten_m
    params.coloring_m = args.g_coloring_m
    params.iter_num = args.g_iter_num
    params.instance_norm = args.g_instance_norm
    params.before_conv = args.g_before_conv

    params.spectral = args.generator_spectral
    params.fully_diff_spectral = args.fully_diff_spectral
    params.spectral_iterations = args.spectral_iterations
    params.conv_singular = args.conv_singular

    params.gan_type = args.gan_type

    params.arch = args.arch
    params.filters_emb = args.filters_emb

    return params


def get_discriminator_params(args):
    params = Namespace()
    params.input_image_shape = args.image_shape
    params.input_cls_shape = (1,)
    if args.arch == 'res':
        if args.dataset == 'tiny-imagenet':
            params.resamples = ("DOWN", "DOWN", "DOWN", "SAME", "SAME")
            params.block_sizes = [args.discriminator_filters / 4, args.discriminator_filters / 2,
                                  args.discriminator_filters,
                                  args.discriminator_filters, args.discriminator_filters]
        elif args.dataset.endswith('imagenet'):
            params.block_sizes = [args.discriminator_filters / 16, args.discriminator_filters / 8,
                                  args.discriminator_filters / 4,
                                  args.discriminator_filters / 2, args.discriminator_filters,
                                  args.discriminator_filters]
            params.resamples = ("DOWN", "DOWN", "DOWN", "DOWN", "DOWN", "SAME")
        else:
            params.block_sizes = tuple([args.discriminator_filters] * 4)
            params.resamples = ('DOWN', "DOWN", "SAME", "SAME")
    else:
        params.block_sizes = [args.discriminator_filters / 8, args.discriminator_filters / 4,
                              args.discriminator_filters / 4, args.discriminator_filters / 2,
                              args.discriminator_filters / 2, args.discriminator_filters,
                              args.discriminator_filters]
        params.resamples = ('SAME', "DOWN", "SAME", "DOWN", "SAME", "DOWN", "SAME")
    params.number_of_classes = 100 if args.dataset == 'cifar100' else (1000 if args.dataset == 'imagenet'
                                                                       else (
        200 if args.dataset == 'tiny-imagenet' else 10))

    params.norm = args.discriminator_norm
    params.coloring = args.discriminator_coloring

    params.decomposition = args.d_decomposition
    params.whitten_m = args.d_whitten_m
    params.coloring_m = args.d_coloring_m
    params.iter_num = args.d_iter_num
    params.instance_norm = args.d_instance_norm
    params.before_conv = args.d_before_conv

    params.spectral = args.discriminator_spectral
    params.fully_diff_spectral = args.fully_diff_spectral
    params.spectral_iterations = args.spectral_iterations
    params.conv_singular = args.conv_singular

    params.type = args.gan_type

    params.sum_pool = args.sum_pool
    params.dropout = args.discriminator_dropout

    params.arch = args.arch
    params.filters_emb = args.filters_emb

    return params


if __name__ == '__main__':
    print()
