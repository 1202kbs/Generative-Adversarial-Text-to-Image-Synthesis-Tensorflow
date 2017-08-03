import tensorflow as tf
import tensorflow.contrib.layers as tcl


class MNIST_Generator(object):

    def __init__(self, X_dim, z_dim, c_dim, e_dim):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.e_dim = e_dim
        self.scope_name = 'g'


    def generate(self, z, phi, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            G_W1 = tf.get_variable('G_W1', [self.e_dim, self.c_dim], initializer=tcl.xavier_initializer())
            G_b1 = tf.get_variable('G_b1', [self.c_dim], initializer=tf.constant_initializer())
            G_W2 = tf.get_variable('G_W2', [self.z_dim + self.c_dim, 128], initializer=tcl.xavier_initializer())
            G_b2 = tf.get_variable('G_b2', [128], initializer=tf.constant_initializer())
            G_W3 = tf.get_variable('G_W3', [128, self.X_dim], initializer=tcl.xavier_initializer())
            G_b3 = tf.get_variable('G_b3', [self.X_dim], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, G_W1) + G_b1)
            code = tf.concat([z, c], axis=1)

            layer1 = tf.nn.relu(tf.matmul(code, G_W2) + G_b2)
            layer2 = tf.nn.sigmoid(tf.matmul(layer1, G_W3) + G_b3)

        return layer2


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)


class MNIST_Discriminator(object):

    def __init__(self, X_dim, z_dim, c_dim, e_dim):
        self.X_dim = X_dim
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.e_dim = e_dim
        self.scope_name = 'd'


    def discriminate(self, X, phi, reuse=False):

        with tf.variable_scope(self.scope_name) as scope:

            if reuse:
                scope.reuse_variables()

            D_W1 = tf.get_variable('D_W1', [self.e_dim, self.c_dim], initializer=tcl.xavier_initializer())
            D_b1 = tf.get_variable('D_b1', [self.c_dim], initializer=tf.constant_initializer())
            D_W2 = tf.get_variable('D_W2', [self.X_dim, 128], initializer=tcl.xavier_initializer())
            D_b2 = tf.get_variable('D_b2', [128], initializer=tf.constant_initializer())
            D_W3 = tf.get_variable('D_W3', [128 + self.c_dim, 1], initializer=tcl.xavier_initializer())
            D_b3 = tf.get_variable('D_b3', [1], initializer=tf.constant_initializer())

            c = tf.nn.sigmoid(tf.matmul(phi, D_W1) + D_b1)
            z = tf.nn.sigmoid(tf.matmul(X, D_W2) + D_b2)
            code = tf.concat([z, c], axis=1)

            layer1 = tf.nn.sigmoid(tf.matmul(code, D_W3) + D_b3)

        return layer1


    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope_name)

