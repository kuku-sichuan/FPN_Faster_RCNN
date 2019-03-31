# # -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
layers = tf.keras.layers


def build_feature_pyramid(backbone_net, config, reuse=tf.AUTO_REUSE):
    """
    :param backbone_net: the dict of network's output at every point.
    :param config: the config of network
    :param reuse:
    :return: the dict of multi-pyramid feature map {P2:,----,P6:}
    """

    backbone_pyramid ={}
    if config.BACKBONE_NET == 'resnet_model':
        backbone_pyramid = backbone_net
    else:
        raise Exception('get no feature maps')
    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid', reuse=reuse):

        feature_pyramid['P5'] = layers.Conv2D(filters=256,
                                              kernel_size=(1, 1),
                                              strides=1,
                                              kernel_initializer="glorot_uniform",
                                              padding="same",
                                              name='C5toP5')(backbone_pyramid['C5'])

        # P6 is down sample of P5
        for layer in range(4, 1, -1):
            p, c = feature_pyramid['P' + str(layer + 1)], backbone_pyramid['C' + str(layer)]
            up_sample = layers.UpSampling2D((2, 2))(p)

            c = layers.Conv2D(filters=256,
                              kernel_size=(1, 1),
                              strides=1,
                              kernel_initializer="glorot_uniform",
                              padding="same",
                              name='C{}toP{}'.format(layer, layer))(c)
            p = up_sample + c
            feature_pyramid['P' + str(layer)] = p
        for layer in range(5, 1, -1):
            p = feature_pyramid['P' + str(layer)]
            p = layers.Conv2D(256, kernel_size=(3, 3), strides=1,
                              kernel_initializer="glorot_uniform",
                              padding="same",
                              name='final_P{}'.format(layer))(p)
            feature_pyramid['P' + str(layer)] = p
        feature_pyramid['P6'] = layers.MaxPool2D(pool_size=(2, 2),
                                                 strides=2, name='final_P6')(feature_pyramid['P5'])

    return feature_pyramid


