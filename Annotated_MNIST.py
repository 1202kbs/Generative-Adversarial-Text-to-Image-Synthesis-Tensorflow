import random

import tensorflow as tf
import numpy as np
import cv2

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)


class Annotated_MNIST():

    def __init__(self, train=False):

        if train:
            self.batches = mnist.train
            self.labels = mnist.train.labels
            self.images = np.reshape(mnist.train.images, [-1, 28, 28])
        else:
            self.batches = mnist.test
            self.labels = mnist.test.labels
            self.images = np.reshape(mnist.test.images, [-1, 28, 28])

        self.nwords = 6
        self.vocab_size = 19
        self.mnist_settings = {0: {'bound': 10, 'skew_range': 60, 'mean': 8, 'line': 14, 'thickness_range': 2}, \
                               1: {'bound': 6, 'skew_range': 25, 'mean': 4, 'line': 20, 'thickness_range': 1},  \
                               2: {'bound': 10, 'skew_range': 60, 'mean': 5, 'line': 11, 'thickness_range': 2}, \
                               3: {'bound': 14, 'skew_range': 20, 'mean': 5, 'line': 18, 'thickness_range': 2}, \
                               4: {'bound': 20, 'skew_range': 50, 'mean': 8, 'line': 10, 'thickness_range': 2}, \
                               5: {'bound': 10, 'skew_range': 30, 'mean': 8, 'line': 20, 'thickness_range': 6}, \
                               6: {'bound': 10, 'skew_range': 25, 'mean': 5, 'line': 7, 'thickness_range': 2},  \
                               7: {'bound': 6, 'skew_range': 20, 'mean': 4, 'line': 22, 'thickness_range': 0},  \
                               8: {'bound': 12, 'skew_range': 25, 'mean': 9, 'line': 10, 'thickness_range': 2}, \
                               9: {'bound': 10, 'skew_range': 30, 'mean': 4, 'line': 22, 'thickness_range': 0}}

        self.word2idx = {'thin': 0, 'normal': 1, 'thick': 2, \
                         'number': 3, \
                         'zero': 4, 'one': 5, 'two': 6, 'three': 7, 'four': 8, 'five': 9, 'six': 10, 'seven': 11, 'eight': 12, 'nine': 13, \
                         'with': 14, \
                         'left': 15, 'average': 16, 'right': 17, \
                         'skew': 18}

        self.idx2word = {idx: word for (word, idx) in self.word2idx.items()}


    def get_nums(self, num):
        idx = np.where(self.labels == num)[0]
        nums = self.images[idx]
        return idx, nums


    def thickness_stats(self, num, line, thickness_range):
        idx, nums = self.get_nums(num)

        m = int(np.mean(np.sum(nums[:, line, :] != 0, axis=1)))
        t_lb = int(m - thickness_range / 2)
        t_ub = int(m + thickness_range / 2)

        l_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) < t_lb)
        n_idx = np.where(np.logical_and(np.sum(nums[:, line, :] != 0, axis=1) >= t_lb, np.sum(nums[:, line, :] != 0, axis=1) <= t_ub))
        h_idx = np.where(np.sum(nums[:, line, :] != 0, axis=1) > t_ub)

        l_idx = idx[l_idx]
        n_idx = idx[n_idx]
        h_idx = idx[h_idx]

        print('Low: {}\nNormal: {}\nHigh: {}\n'.format(len(l_idx), len(n_idx), len(h_idx)))

        return l_idx, n_idx, h_idx


    def get_thickness(self, img, mean, line, thickness_range):
        t_lb = int(mean - thickness_range / 2)
        t_ub = int(mean + thickness_range / 2)

        if np.sum(img[line, :] != 0) < t_lb:
            return 'thin'
        elif np.logical_and(np.sum(img[line, :] != 0) >= t_lb, np.sum(img[line, :] != 0) <= t_ub):
            return 'normal'
        else:
            return 'thick'


    def __rotate_and_scale(self, img, angle, scale):
        M = cv2.getRotationMatrix2D((14, 14), angle, scale)
        return cv2.warpAffine(img, M, (28, 28))


    def skew_stats(self, num, bound, skew_range):
        lb = int(14 - bound / 2)
        ub = int(14 + bound / 2)
        s_lb = -int(skew_range / 2)
        s_ub = int(skew_range / 2)

        idx, nums = self.get_nums(num)

        skews = []
        for num in nums:

            li = []
            max_overlap = 0
            max_angle = 0
            for angle in range(-90, 90, 5):
                temp1 = self.__rotate_and_scale(num, angle, 1)
                temp2 = temp1[:, lb:ub]

                if max_overlap <= np.sum(temp2 != 0):
                    max_overlap = np.sum(temp2 != 0)
                    max_angle = angle

            skews.append(max_angle)

        skews = np.array(skews)
        l_idx = np.where(skews < s_lb)
        n_idx = np.where(np.logical_and(skews >= s_lb, skews <= s_ub))
        h_idx = np.where(skews > s_ub)

        l_idx = idx[l_idx]
        n_idx = idx[n_idx]
        h_idx = idx[h_idx]

        print('Low: {}\nNormal: {}\nHigh: {}\n'.format(len(l_idx), len(n_idx), len(h_idx)))

        return l_idx, n_idx, h_idx


    def get_skew(self, img, bound, skew_range):
        lb = int(14 - bound / 2)
        ub = int(14 + bound / 2)
        s_lb = -int(skew_range / 2)
        s_ub = int(skew_range / 2)

        max_overlap = 0
        max_angle = 0
        for angle in range(-90, 90, 5):
            temp1 = self.__rotate_and_scale(img, angle, 1)
            temp2 = temp1[:, lb:ub]

            if max_overlap <= np.sum(temp2 != 0):
                max_overlap = np.sum(temp2 != 0)
                max_angle = angle

        if max_angle < s_lb:
            return 'left skew'
        elif np.logical_and(max_angle >= s_lb, max_angle <= s_ub):
            return 'average skew'
        else:
            return 'right skew'


    def get_description(self, img, num):

        settings = self.mnist_settings[num]
        bound = settings['bound']
        skew_range = settings['skew_range']
        mean = settings['mean']
        line = settings['line']
        thickness_range = settings['thickness_range']

        skew = self.get_skew(img, bound, skew_range)
        thickness = self.get_thickness(img, mean, line, thickness_range)

        if num == 0:
            number = 'zero'
        elif num == 1:
            number = 'one'
        elif num == 2:
            number = 'two'
        elif num == 3:
            number = 'three'
        elif num == 4:
            number = 'four'
        elif num == 5:
            number = 'five'
        elif num == 6:
            number = 'six'
        elif num == 7:
            number = 'seven'
        elif num == 8:
            number = 'eight'
        else:
            number = 'nine'

        res = thickness + ' number ' + number + ' with ' + skew

        return res


    def next_batch(self, batch_size, resize=False, convert_to_idx=True):
        batch_xs, batch_ys = self.batches.next_batch(batch_size)

        images = np.reshape(batch_xs, [-1, 28, 28])
        labels = batch_ys

        descriptions = []

        for i in range(batch_size):
            img = images[i]
            num = labels[i]
            description = self.get_description(img, num).split()

            if convert_to_idx:
                temp = list(map(self.word2idx.get, description))
                descriptions.append(temp)
            else:
                descriptions.append(description)

        if resize:

            batch_xs = np.reshape(images, [-1, 784])
            images_small = []

            for i in range(batch_size):
                images_small.append(cv2.resize(images[i], (14, 14), interpolation=0))

            images_small = np.array(images_small)
            batch_xs_small = np.reshape(images_small, [-1, 196])
            batch_ys = labels

            return descriptions, batch_xs, batch_xs_small, batch_ys
        
        else:
            batch_xs = np.reshape(images, [-1, 784])
            batch_ys = labels

            return descriptions, batch_xs, batch_ys


    def generate_sentences(self, num, divide=True, convert_to_idx=True):
        sentences = []

        thickness = ['thin', 'normal', 'thick']
        nums = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        skew = ['left skew', 'average skew', 'right skew']

        for i in range(num):
            sentence = ''

            sentence += random.choice(thickness)
            sentence += ' number '
            sentence += random.choice(nums)
            sentence += ' with '
            sentence += random.choice(skew)

            if divide:
                sentences.append(sentence.split())
            else:
                sentences.append(sentence)

        if convert_to_idx:

            for i in range(num):
                sentences[i] = list(map(self.word2idx.get, sentences[i]))

            return sentences
        
        else:

            return sentences


    def convert_to_word(self, sentences, concat=True):
        res = []

        for sentence in sentences:
            res.append(list(map(self.idx2word.get, sentence)))

        if concat:

            for i in range(len(res)):

                sent = ''
                for word in res[i]:
                    sent += word + ' '

                res[i] = str.rstrip(sent)

        return res


    def convert_to_idx(self, sentences):
        res = []

        for sentence in sentences:
            res.append(list(map(self.word2idx.get, sentence.split())))

        return res
