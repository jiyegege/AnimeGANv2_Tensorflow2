from glob import glob

import keras.layers
import yaml
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from net.discriminator import Discriminator
from net.generator import Generator
from tools.data_loader import ImageGenerator
from tools.ops import *
from tools.utils import *


class AnimeGANv2(object):
    def __init__(self, args):
        if args.hyperparameters.lower() == 'true':
            self.hyperparameters = True
        else:
            self.hyperparameters = False

        config_dict = yaml.safe_load(open(args.config_path, 'r'))
        # Initialize a new wandb run
        wandb.init(project="AnimeGANv2_Tensorflow2", entity="roger_ds", sync_tensorboard=True, config=config_dict)

        self.model_name = 'AnimeGANv2'
        self.checkpoint_dir = args.checkpoint_dir
        self.log_dir = args.log_dir
        self.dataset_name = args.dataset
        self.save_freq = args.save_freq

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        """ Discriminator """
        self.n_dis = args.n_dis
        self.ch = args.ch

        self.sample_dir = os.path.join(args.sample_dir, self.model_dir)
        check_folder(self.sample_dir)

        self.real_image_generator = ImageGenerator('./dataset/train_photo', self.img_size, wandb.config.batch_size)
        self.anime_image_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/style'), self.img_size,
                                                    wandb.config.batch_size)
        self.anime_smooth_generator = ImageGenerator('./dataset/{}'.format(self.dataset_name + '/smooth'),
                                                     self.img_size, wandb.config.batch_size)
        self.dataset_num = max(self.real_image_generator.num_images, self.anime_image_generator.num_images)

        self.p_model = local_variables_init()

        self.pre_train_weight = args.pre_train_weight

        print()
        print("##### Information #####")
        print("# gan type : ", wandb.config.gan_type)
        print("# dataset : ", self.dataset_name)
        print("# max dataset number : ", self.dataset_num)
        print("# batch_size : ", wandb.config.batch_size)
        print("# epoch : ", wandb.config.epoch)
        print("# init_epoch : ", wandb.config.init_epoch)
        print("# training image size [H, W] : ", self.img_size)
        print("# g_adv_weight,d_adv_weight,con_weight,sty_weight,color_weight,tv_weight : ", wandb.config.g_adv_weight,
              wandb.config.d_adv_weight, wandb.config.con_weight, wandb.config.sty_weight, wandb.config.color_weight,
              wandb.config.tv_weight)
        print("# init_lr,g_lr,d_lr : ", wandb.config.init_lr, wandb.config.g_lr, wandb.config.d_lr)
        print(f"# training_rate G -- D: {wandb.config.training_rate} : 1")
        print()

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self):
        G = Generator()
        G.build(input_shape=(None, self.img_size[0], self.img_size[1], self.img_ch))
        # G.summary()
        return G

    ##################################################################################
    # Discriminator
    ##################################################################################

    def discriminator(self):
        D = Discriminator(self.ch, self.n_dis, wandb.config.sn)
        D.build(input_shape=(None, self.img_size[0], self.img_size[1], self.img_ch))
        # D.summary()
        return D

    ##################################################################################
    # Model
    ##################################################################################
    @tf.function
    def gradient_panalty(self, real, fake, discriminator):
        if wandb.config.gan_type.__contains__('dragan'):
            eps = tf.random.uniform(shape=tf.shape(real), minval=0., maxval=1.)
            _, x_var = tf.nn.moments(real, axes=[0, 1, 2, 3])
            x_std = tf.sqrt(x_var)  # magnitude of noise decides the size of local region

            fake = real + 0.5 * x_std * eps

        alpha = tf.random.uniform(shape=[wandb.config.batch_size, 1, 1, 1], minval=0., maxval=1.)
        interpolated = real + alpha * (fake - real)
        logit = discriminator(interpolated)
        # gradient of D(interpolated)
        grad = tf.gradients(logit, interpolated)[0]
        grad_norm = tf.norm(keras.layers.Flatten()(grad), axis=1)  # l2 norm

        GP = 0
        # WGAN - LP
        if wandb.config.gan_type.__contains__('lp'):
            GP = wandb.config.ld * tf.reduce_mean(tf.square(tf.maximum(0.0, grad_norm - 1.)))

        elif wandb.config.gan_type.__contains__('gp') or wandb.config.gan_type == 'dragan':
            GP = wandb.config.ld * tf.reduce_mean(tf.square(grad_norm - 1.))

        return GP

    def train(self):

        """ Input Image"""
        real_img_op, anime_img_op, anime_smooth_op = self.real_image_generator.load_images(), \
                                                     self.anime_image_generator.load_images(), \
                                                     self.anime_smooth_generator.load_images()

        # real, anime, anime_gray, anime_smooth = real_img_op, anime_img_op, \
        #                                         anime_img_op, anime_smooth_op
        """ Define Generator, Discriminator """
        generated = self.generator()
        discriminator = self.discriminator()

        # summary writer
        self.writer = tf.summary.create_file_writer(self.log_dir + '/' + self.model_dir)

        """ Training """

        init_optim = Adam(wandb.config.init_lr, beta_1=0.5, beta_2=0.999)
        G_optim = Adam(wandb.config.g_lr, beta_1=0.5, beta_2=0.999)
        D_optim = Adam(wandb.config.d_lr, beta_1=0.5, beta_2=0.999)

        # saver to save model
        self.saver = tf.train.Checkpoint(generated=generated, discriminator=discriminator, p_model=self.p_model,
                                         G_optim=G_optim, D_optim=D_optim)

        # restore check-point if it exits
        if self.pre_train_weight:
            self.load_pre_weight(self.pre_train_weight)
            start_epoch = 0
            print("Load pre-trained weight Success!")
        else:
            could_load, checkpoint_counter = self.load(self.checkpoint_dir)
            if could_load:
                start_epoch = checkpoint_counter + 1

                print(" [*] Load SUCCESS")
            else:
                start_epoch = 0

                print(" [!] Load failed...")

        init_mean_loss = []
        mean_loss = []
        j = wandb.config.training_rate
        for epoch in range(start_epoch, wandb.config.epoch):
            total_step = int(self.dataset_num / wandb.config.batch_size)
            with tqdm(range(total_step)) as tbar:
                for step in range(total_step):
                    real = next(real_img_op)[0]
                    anime = next(anime_img_op)[0]
                    anime_gray = next(anime_img_op)[1]
                    anime_smooth = next(anime_smooth_op)[1]

                    if epoch < wandb.config.init_epoch:
                        init_loss = self.init_train_step(generated, init_optim, epoch, real)
                        init_mean_loss.append(init_loss)
                        tbar.set_description('Epoch %d' % epoch)
                        tbar.set_postfix(init_v_loss=init_loss.numpy(), mean_v_loss=np.mean(init_mean_loss))
                        tbar.update()
                        if (step + 1) % 200 == 0:
                            init_mean_loss.clear()
                    else:
                        if j == wandb.config.training_rate:
                            # Update D network
                            d_loss = self.d_train_step(real, anime, anime_gray, anime_smooth,
                                                       generated, discriminator, D_optim, epoch)

                        # Update G network
                        g_loss = self.g_train_step(real, anime_gray, generated, discriminator, G_optim, epoch)

                        mean_loss.append([d_loss, g_loss])
                        tbar.set_description('Epoch %d' % epoch)
                        if j == wandb.config.training_rate:
                            tbar.set_postfix(d_loss=d_loss.numpy(), g_loss=g_loss.numpy(),
                                             mean_d_loss=np.mean(mean_loss, axis=0)[0],
                                             mean_g_loss=np.mean(mean_loss, axis=0)[1])
                        else:
                            tbar.set_postfix(g_loss=g_loss.numpy(), mean_g_loss=np.mean(mean_loss, axis=0)[1])
                        tbar.update()

                        if (step + 1) % 200 == 0:
                            mean_loss.clear()

                        j = j - 1
                        if j < 1:
                            j = wandb.config.training_rate
            if not self.hyperparameters:
                if (epoch + 1) >= wandb.config.init_epoch and np.mod(epoch + 1, self.save_freq) == 0:
                    self.save(self.checkpoint_dir, epoch)

            if epoch >= wandb.config.init_epoch - 1:
                """ Result Image """
                val_files = glob('./dataset/{}/*.*'.format('val'))
                # save_path = './{}/{:03d}/'.format(self.sample_dir, epoch)
                # check_folder(save_path)
                val_images = []
                for i, sample_file in enumerate(val_files):
                    # print('val: ' + str(i) + sample_file)
                    sample_image = np.asarray(load_test_data(sample_file, self.img_size))
                    test_real = sample_image
                    test_generated_predict = generated.predict(test_real)
                    # save_images(test_real, save_path + '{:03d}_a.jpg'.format(i), None)
                    # save_images(test_generated_predict, save_path + '{:03d}_b.jpg'.format(i), None)
                    if i == 0 or i == 26 or i == 5:
                        val_images.append(
                            wandb.Image(test_generated_predict, caption="Name:{}, epoch:{}".format(i, epoch)))
                        # with self.writer.as_default(step=epoch):
                        #     """" Summary """
                        #     tf.summary.image(name='val_data_' + str(i), data=test_generated_predict, step=epoch)
                wandb.log({'val_data': val_images}, step=epoch)
                if not self.hyperparameters:
                    save_model_path = 'save_model'
                    if not os.path.exists(save_model_path):
                        os.makedirs(save_model_path)
                    generated.save(os.path.join(save_model_path, 'generated_' + self.dataset_name), save_format='tf')

    @tf.function
    def init_train_step(self, generated, init_optim, epoch, real):
        with tf.GradientTape() as tape:
            generator_images = generated(real)
            # init pharse
            init_c_loss = con_loss(self.p_model, real, generator_images)
            init_loss = wandb.config.con_weight * init_c_loss
        grads = tape.gradient(init_loss, generated.trainable_variables)
        init_optim.apply_gradients(zip(grads, generated.trainable_variables))
        # wandb.log("G_init", init_loss.numpy(), step=epoch)
        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar(name='G_init_loss', data=init_loss)

        return init_loss

    @tf.function
    def g_train_step(self, real, anime_gray, generated,
                     discriminator, G_optim, epoch):
        with tf.GradientTape() as tape:
            fake_image = generated(real)
            generated_logit = discriminator(fake_image)
            # gan
            c_loss, s_loss = con_sty_loss(self.p_model, real, anime_gray, fake_image)
            tv_loss = wandb.config.tv_weight * total_variation_loss(fake_image)
            col_loss = color_loss(real, fake_image)
            t_loss = wandb.config.con_weight * c_loss + wandb.config.sty_weight * s_loss + col_loss * wandb.config.color_weight + tv_loss

            g_loss = wandb.config.g_adv_weight * generator_loss(wandb.config.gan_type, generated_logit)
            Generator_loss = t_loss + g_loss

        grads = tape.gradient(Generator_loss, generated.trainable_variables)
        G_optim.apply_gradients(zip(grads, generated.trainable_variables))

        # wandb.log("Generator_loss", Generator_loss.numpy(), step=epoch)
        # wandb.log("G_con_loss", c_loss.numpy(), step=epoch)
        # wandb.log("G_sty_loss", s_loss.numpy(), step=epoch)
        # wandb.log("G_color_loss", col_loss.numpy(), step=epoch)
        # wandb.log("G_gan_loss", g_loss.numpy(), step=epoch)
        # wandb.log("G_pre_model_loss", t_loss.numpy(), step=epoch)
        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar("Generator_loss", Generator_loss)
            tf.summary.scalar("G_con_loss", c_loss)
            tf.summary.scalar("G_sty_loss", s_loss)
            tf.summary.scalar("G_color_loss", col_loss)

            tf.summary.scalar("G_gan_loss", g_loss)
            tf.summary.scalar("G_pre_model_loss", t_loss)
        return Generator_loss

    @tf.function
    def d_train_step(self, real, anime, anime_gray, anime_smooth, generated, discriminator,
                     D_optim, epoch):

        with tf.GradientTape() as tape:
            fake_image = generated(real)
            d_anime_logit = discriminator(anime)
            d_anime_gray_logit = discriminator(anime_gray)
            d_smooth_logit = discriminator(anime_smooth)
            generated_logit = discriminator(fake_image)
            """ Define Loss """
            if wandb.config.gan_type.__contains__('gp') or wandb.config.gan_type.__contains__('lp') or \
                    wandb.config.gan_type.__contains__('dragan'):
                GP = self.gradient_panalty(real=anime, fake=fake_image, discriminator=discriminator)
            else:
                GP = 0.0

            d_loss = wandb.config.d_adv_weight * discriminator_loss(wandb.config.gan_type, d_anime_logit,
                                                                   d_anime_gray_logit,
                                                                   generated_logit,
                                                                   d_smooth_logit, epoch, self.writer,
                                                                   wandb.config.real_loss_weight,
                                                                   wandb.config.fake_loss_weight,
                                                                   wandb.config.gray_loss_weight,
                                                                   wandb.config.real_blur_loss_weight) + GP
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        D_optim.apply_gradients(zip(grads, discriminator.trainable_variables))

        with self.writer.as_default(step=epoch):
            """" Summary """
            tf.summary.scalar("Discriminator_loss", d_loss)

        return d_loss

    @property
    def model_dir(self):
        return "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(self.model_name, self.dataset_name,
                                                   wandb.config.gan_type,
                                                   int(wandb.config.g_adv_weight), int(wandb.config.d_adv_weight),
                                                   int(wandb.config.con_weight), int(wandb.config.sty_weight),
                                                   int(wandb.config.color_weight), int(wandb.config.tv_weight))

    def save(self, checkpoint_dir, epoch):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        ckpt_manager = tf.train.CheckpointManager(self.saver, checkpoint_dir,
                                                  max_to_keep=5)
        ckpt_manager.save(checkpoint_number=epoch)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            self.saver.restore(os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def load_pre_weight(self, checkpoint_dir):
        print(" [*] Reading load pre train checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)  # checkpoint file information

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)  # first line
            self.saver.restore(os.path.join(checkpoint_dir, ckpt_name))
            print(" [*] Success to read {}".format(os.path.join(checkpoint_dir, ckpt_name)))
