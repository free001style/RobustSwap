from argparse import ArgumentParser


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument('--exp_dir', type=str, default="running_exp",
                                 help='Path to experiment output directory')
        # ================= Model =====================
        self.parser.add_argument('--out_size', type=int, default=1024, help='output image size')
        # ================= Dataset =====================
        self.parser.add_argument('--celeba_dataset_root', default='../datasets/CelebAMask-HQ', type=str,
                                 help='CelebAMask-HQ dataset root path')
        self.parser.add_argument('--ffhq_dataset_root', default='/home/aalanov/Bobkov_Denis/datasets/FFHQ', type=str,
                                 help='FFHQ dataset root path')
        self.parser.add_argument('--dataset_name', default='ffhq', type=str, help='which dataset to use')
        self.parser.add_argument('--flip_p', default=0.5, type=float, help='probability to apply horizontal flipping')
        self.parser.add_argument('--ds_frac', default=1.0, type=float, help='dataset fraction')
        self.parser.add_argument('--batch_size', default=8, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=8, type=int, help='Batch size for testing and inference')
        self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=4, type=int,
                                 help='Number of test/inference dataloader workers')

        # ================= Training =====================
        self.parser.add_argument('--learning_rate', default=1e-4, type=float, help='Optimizer learning rate')
        self.parser.add_argument('--optim_name', default='adam', type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_G', default=True, type=bool, help='Whether to train the styleGAN model')
        self.parser.add_argument('--train_D', default=True, type=bool,
                                 help='Whether to train the styleGAN Discriminator')
        self.parser.add_argument('--dist_train', default=True, type=bool, help='distributed training')
        self.parser.add_argument('--local_rank', default=-1, type=int, help='local rank for distributed training')
        self.parser.add_argument('--d_reg_every', default=16, type=int,
                                 help='interval of the applying r1 regularization')
        self.parser.add_argument('--d_every', default=15, type=int, help='interval of the updating discriminator')

        self.parser.add_argument('--max_steps', default=500000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=500, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50, type=int,
                                 help='Interval for logging metrics to wandb')
        self.parser.add_argument('--val_interval', default=5000 * 2 * 2, type=int, help='Validation interval')
        self.parser.add_argument('--save_interval', default=5000, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--same_image_interval', default=3, type=int,
                                 help='Interval for sampling same source and target')

        # ================= Loss Function =====================
        self.parser.add_argument('--recon_lambda', default=1.0, type=float,
                                 help='Reconstruction loss multiplier factor')
        self.parser.add_argument('--id_lambda', default=1.0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument('--id_loss_multiscale', default=True, type=bool,
                                 help='Whether to apply multi scale in ID loss')
        self.parser.add_argument('--pl_lambda', default=100.0, type=float, help='PL loss multiplier factor')
        self.parser.add_argument('--r1_lambda', default=10, type=float, help='R1 regularization loss multiplier factor')
        self.parser.add_argument('--g_adv_lambda', default=0.01, type=float,
                                 help='generator adversarial loss multiplier factor')

        # ================== styleGAN2 ==================
        self.parser.add_argument('--stylegan_weights',
                                 default='./pretrained_ckpts/stylegan2/stylegan2-ffhq-config-f.pt',
                                 type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--learn_in_w', default=False, help='Whether to learn in w space instead of w+')
        self.parser.add_argument('--start_from_latent_avg', default=True,
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument('--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument('--n_styles', default=18, type=int, help='StyleGAN layers number')

        # auxiliary models
        self.parser.add_argument('--ir_se50_path', default='./pretrained_ckpts/auxiliary/model_ir_se50.pth', type=str,
                                 help='Path to ir_se50 model weights')
        self.parser.add_argument('--arcface_path', default='./pretrained_ckpts/auxiliary/iresnet50-7f187506.pth',
                                 type=str,
                                 help='Path to iresnet50 model weights')
        self.parser.add_argument('--sfe_inverter_path', default='./pretrained_ckpts/inverter/sfe_inverter.pt',
                                 type=str,
                                 help='Path to sfe')
        self.parser.add_argument('--deca_path', default='./pretrained_ckpts/deca/deca_model.tar', type=str,
                                 help='Path to Deca')
        self.parser.add_argument('--flame_path', default='./pretrained_ckpts/flame/generic_model.pkl', type=str,
                                 help='Path to flame')
        self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint')

    def parse(self):
        opts = self.parser.parse_args()
        return opts
