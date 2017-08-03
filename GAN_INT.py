import os

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from ops2 import MNIST_Generator, MNIST_Discriminator
from Annotated_MNIST import Annotated_MNIST
from utils import plot


class GAN_INT(object):

    def __init__(self, config, sess):
        self.X_dim = config.X_dim
        self.nwords = config.nwords
        self.vocab_size = config.vocab_size
        self.z_dim = config.z_dim
        self.c_dim = config.c_dim
        self.e_dim = config.e_dim
        self.d_update = config.d_update
        self.niter = config.niter
        self.batch_size = config.batch_size
        self.beta = config.beta
        self.lr = config.lr
        self.use_adam = config.use_adam

        if self.use_adam:
            self.optimizer = tf.train.AdamOptimizer
        else:
            self.optimizer = tf.train.GradientDescentOptimizer

        self.checkpoint_dir = config.checkpoint_dir
        self.image_dir = config.image_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)

        self.annotator = Annotated_MNIST(train=True)

        self.X = tf.placeholder(tf.float32, [None, self.X_dim], 'X')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], 'z')
        self.context_r = tf.placeholder(tf.int32, [None, self.nwords], 'context_r')
        self.context_f = tf.placeholder(tf.int32, [None, self.nwords], 'context_f')

        self.sess = sess


    def z_sampler(self, dim1, dim2):
        return np.random.uniform(-1, 1, size=[dim1, dim2])


    def build_model(self):
        E = tf.Variable(tf.random_normal([self.vocab_size, self.e_dim]))
        phi_r = tf.nn.embedding_lookup(E, self.context_r)
        phi_r = tf.reduce_sum(phi_r, axis=1)

        phi_f = tf.nn.embedding_lookup(E, self.context_f)
        phi_f = tf.reduce_sum(phi_f, axis=1)

        phi_i = phi_r * self.beta + phi_f * (1 - self.beta)

        generator = MNIST_Generator(self.X_dim, self.z_dim, self.c_dim, self.e_dim)
        discriminator = MNIST_Discriminator(self.X_dim, self.z_dim, self.c_dim, self.e_dim)

        self.G_r = generator.generate(self.z, phi_r, reuse=False)
        self.G_i = generator.generate(self.z, phi_i, reuse=True)

        #    D_imge_text
        self.D_fake_real = discriminator.discriminate(self.G_r, phi_r, reuse=False)
        self.D_fake_fake = discriminator.discriminate(self.G_i, phi_i, reuse=True)
        self.D_real_real = discriminator.discriminate(self.X, phi_r, reuse=True)
        self.D_real_fake = discriminator.discriminate(self.X, phi_f, reuse=True)

        gaussian_smp = tf.abs(tf.random_normal([]))
        
        self.G_loss = -tf.reduce_mean(tf.log(self.D_fake_real + 1e-10) + tf.log(self.D_fake_fake + 1e-10))
        self.D_loss = -tf.reduce_mean(tf.log(self.D_real_real + 1e-10) + (tf.log(1 - self.D_fake_real + 1e-10) + tf.log(1 - self.D_real_fake + 1e-10)) / 2)
        
        G_params = generator.vars
        D_params = discriminator.vars

        G_optimizer = self.optimizer(self.lr)
        self.G_optim = G_optimizer.minimize(loss=self.G_loss, var_list=G_params + [E])

        D_optimizer = self.optimizer(self.lr)
        self.D_optim = D_optimizer.minimize(loss=self.D_loss, var_list=D_params + [E])

        tf.global_variables_initializer().run()
        self.saver = tf.train.Saver(var_list=G_params + D_params + [E])


    def train(self):
   
        descriptions, batch_xs, batch_ys = self.annotator.next_batch(self.batch_size, resize=False, convert_to_idx=True)
        fake_context = self.annotator.generate_sentences(self.batch_size, divide=True, convert_to_idx=True)

        feed_dict = {self.X: batch_xs, self.context_r: descriptions, self.context_f: fake_context, self.z: self.z_sampler(self.batch_size, self.z_dim)}

        for _ in range(self.d_update):
            _, D_loss = self.sess.run([self.D_optim, self.D_loss], feed_dict=feed_dict)
        _, G_loss = self.sess.run([self.G_optim, self.G_loss], feed_dict=feed_dict)

        return G_loss, D_loss


    def run(self):

        for iteration in range(self.niter):
            avg_G_loss, avg_D_loss = self.train()

            if iteration % 200 == 0:
                state = {'G Loss': avg_G_loss, 'D Loss': avg_D_loss, 'Iteration': iteration}
                print(state)

                context = self.annotator.generate_sentences(16, divide=True, convert_to_idx=True)
                descs = self.annotator.convert_to_word(context, concat=True)
                print(descs)
                
                samples = self.sess.run(self.G_r, feed_dict={self.context_r: context, self.z: self.z_sampler(16, self.z_dim)})
                fig = plot(samples, [28, 28])
                plt.savefig(os.path.join(self.image_dir, '{:05d}.png'.format(iteration)), bbox_inches='tight')
                plt.close(fig)

                self.saver.save(self.sess, os.path.join(self.checkpoint_dir, 'StackGAN.model'))


    def generate(self, sentences):
        self.load()

        num_instances = len(sentences)
        context = self.annotator.convert_to_idx(sentences)
        return self.sess.run(self.G_r, feed_dict={self.context_r: context, self.z: self.z_sampler(num_instances, self.z_dim)})


    def load(self):

        print('[*] Reading Checkpoints...')

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise Exception('[!] No Checkpoints Found')
