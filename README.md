StackGAN in Tensorflow
======================

Tensorflow implementation of [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396) for MNIST handwritten digit dataset.

Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/) and [OpenCV](http://opencv.org). The MNIST dataset is stored in the 'MNIST_data' directory. The files will be automatically downloaded if the dataset does not exist.
    
If you want to use `--show_progress True` option, you need to install python package `progress`.

    $ pip install progress

Usage
-----

To train a vanilla GAN (GAN_CLS) with z dimension 20 and generator input code dimension 2, run the following command:

    $ python GAN_CLS_main.py --z_dim 20 --c_dim 2

To see all training options, run:

    $ python GAN_CLS_main.py --help

which will print:

    usage: GAN_CLS_main.py [-h] [--X_dim X_DIM] [--nwords NWORDS]
                       [--vocab_size VOCAB_SIZE] [--z_dim Z_DIM]
                       [--c_dim C_DIM] [--e_dim E_DIM] [--d_update D_UPDATE]
                       [--niter NITER] [--batch_size BATCH_SIZE] [--lr LR]
                       [--checkpoint_dir CHECKPOINT_DIR]
                       [--image_dir IMAGE_DIR] [--use_adam [USE_ADAM]]
                       [--nouse_adam]

    optional arguments:
      -h, --help            show this help message and exit
      --X_dim X_DIM         dimension of the original image [784]
      --nwords NWORDS       number of words in the input sentence (e.g. "thin
                            number one with left skew") [6]
      --vocab_size VOCAB_SIZE
                            size of the vocabulary [19]
      --z_dim Z_DIM         dimension of the generator input noise variable z [20]
      --c_dim C_DIM         dimension of input code variable c [2]
      --e_dim E_DIM         dimension of the word embedding phi [20]
      --d_update D_UPDATE   update the discriminator weights [d_update] times per
                            generator update [5]
      --niter NITER         number of iterations to use during training [5500]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --lr LR               learning rate of the optimizer to use during training
                            [0.001]
      --checkpoint_dir CHECKPOINT_DIR
                            checkpoint directory [./checkpoints]
      --image_dir IMAGE_DIR
                            directory to save generated images to [./images]
      --use_adam [USE_ADAM]
                            if True, use Adam optimizer; otherwise, use SGD [True]
      --nouse_adam


To train a GAN with manifold interpolation (GAN-INT) with z dimension 20 and generator code input dimension 2, run the following command:

    $ python GAN_INT_main.py --z_dim 20 --c_dim 2

To see all training options, run:

    $ python GAN_INT_main.py --help

which will print:

    usage: GAN_INT_main.py [-h] [--X_dim X_DIM] [--nwords NWORDS]
                       [--vocab_size VOCAB_SIZE] [--z_dim Z_DIM]
                       [--c_dim C_DIM] [--e_dim E_DIM] [--d_update D_UPDATE]
                       [--niter NITER] [--batch_size BATCH_SIZE] [--beta BETA]
                       [--lr LR] [--checkpoint_dir CHECKPOINT_DIR]
                       [--image_dir IMAGE_DIR] [--use_adam [USE_ADAM]]
                       [--nouse_adam]

    optional arguments:
      -h, --help            show this help message and exit
      --X_dim X_DIM         dimension of the original image [784]
      --nwords NWORDS       number of words in the input sentence (e.g. "thin
                            number one with left skew") [6]
      --vocab_size VOCAB_SIZE
                            size of the vocabulary [19]
      --z_dim Z_DIM         dimension of the generator input noise variable z [20]
      --c_dim C_DIM         dimension of input code variable c [2]
      --e_dim E_DIM         dimension of the word embedding phi [20]
      --d_update D_UPDATE   update the discriminator weights [d_update] times per
                            generator update [1]
      --niter NITER         number of epochs to use during training [5500]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --beta BETA           variable that parameterizes the amount of
                            interpolation between two text embeddings [0.5]
      --lr LR               learning rate of the optimizer to use during training
                            [0.001]
      --checkpoint_dir CHECKPOINT_DIR
                            checkpoint directory [./checkpoints]
      --image_dir IMAGE_DIR
                            directory to save generated images to [./images]
      --use_adam [USE_ADAM]
                            if True, use Adam optimizer; otherwise, use SGD [True]
      --nouse_adam

Notes
-----

The Annotated_MNIST.py is a thickness and skew labeler for MNIST handwritten digit dataset, intended to be used for toy text to image generation tasks or specific classification tasks. More details can be found in [this](https://github.com/1202kbs/Annotated_MNIST) repository.