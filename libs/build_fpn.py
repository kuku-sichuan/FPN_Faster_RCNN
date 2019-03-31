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
    if config.BACKBONE_NET == 'resnet_v2_50':
        backbone_pyramid = backbone_net
    else:
        raise Exception('get no feature maps')
    feature_pyramid = {}
    with tf.variable_scope('build_feature_pyramid', reuse=reuse):

        feature_pyramid['P5'] = layers.Conv2D(filters=256,
                                              kernel_size=(1, 1),
                                              stride=1,
                                              name='C5toP5')(backbone_pyramid['C5'])

        # P6 is down sample of P5
        for layer in range(4, 1, -1):
            p, c = feature_pyramid['P' + str(layer + 1)], backbone_pyramid['C' + str(layer)]
            up_sample = layers.UpSampling2D((2, 2), data_format=config.DATA_FORMAT)(p)

            c = layers.Conv2D(filters=256,
                              kernel_size=(1, 1),
                              stride=1,
                              name='C{}toP{}'.format(layer, layer))(c)
            p = up_sample + c
            feature_pyramid['P' + str(layer)] = p
        for layer in range(5, 1, -1):
            p = feature_pyramid['P' + str(layer)]
            p = layers.Conv2D(256, kernel_size=(3, 3), stride=1,
                              padding='SAME',
                              name='final_P{}'.format(layer))(p)
            feature_pyramid['P' + str(layer)] = p
        feature_pyramid['P6'] = layers.MaxPool2D(feature_pyramid['P5'],
                                                 kernel_size=(2, 2),
                                                 stride=2, name='final_P6')

    return feature_pyramid


