import os

import tensorflow as tf
import pprint

from GAN_INT import GAN_INT

pp = pprint.PrettyPrinter()

flags = tf.app.flags

flags.DEFINE_integer('X_dim', 784, 'dimension of the original image [784]')
flags.DEFINE_integer('nwords', 6, 'number of words in the input sentence (e.g. "thin number one with left skew") [6]')
flags.DEFINE_integer('vocab_size', 19, 'size of the vocabulary [19]')
flags.DEFINE_integer('z_dim', 20, 'dimension of the generator input noise variable z [20]')
flags.DEFINE_integer('c_dim', 2, 'dimension of input code variable c [2]')
flags.DEFINE_integer('e_dim', 20, 'dimension of the word embedding phi [20]')
flags.DEFINE_integer('d_update', 1, 'update the discriminator weights [d_update] times per generator update [1]')
flags.DEFINE_integer('niter', 5500, 'number of epochs to use during training [5500]')
flags.DEFINE_integer('batch_size', 128, 'batch size to use during training [128]')
flags.DEFINE_float('beta', 0.5, 'variable that parameterizes the amount of interpolation between two text embeddings [0.5]')
flags.DEFINE_float('lr', 0.001, 'learning rate of the optimizer to use during training [0.001]')
flags.DEFINE_string('checkpoint_dir', './checkpoints', 'checkpoint directory [./checkpoints]')
flags.DEFINE_string('image_dir', './images', 'directory to save generated images to [./images]')
flags.DEFINE_bool('use_adam', True, 'if True, use Adam optimizer; otherwise, use SGD [True]')

FLAGS = flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    pp.pprint(flags.FLAGS.__flags)

    with tf.Session() as sess:
        model = GAN_INT(FLAGS, sess)
        model.build_model()
        model.run()

if __name__ == '__main__':
    tf.app.run()