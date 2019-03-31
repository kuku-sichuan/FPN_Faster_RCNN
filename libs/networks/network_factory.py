from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from . import resnet


def get_network_byname(inputs,
                       config,
                       is_training,
                       reuse):

    if config.BACKBONE_NET == 'resnet_model':
        features_map = resnet.resnet_v2(inputs=inputs,
                                        training=is_training,
                                        reuse=reuse)
        return None, features_map


