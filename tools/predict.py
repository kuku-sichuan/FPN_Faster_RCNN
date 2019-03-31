# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import tensorflow as tf
from libs.networks.network_factory import get_network_byname
from libs import build_rpn, build_head, build_fpn
sys.path.append('../')


def model_fn(features,
             mode,
             params):
    # ***********************************************************************************************
    # *                                        Backbone Net                                           *
    # ***********************************************************************************************
    net_config = params["net_config"]
    IS_TRAINING = False

    image_batch = features["image"]
    image_window = features["image_window"]
    origin_image_batch = image_batch + tf.convert_to_tensor(net_config.PIXEL_MEANS)
    # there is is_training means that bn is training, so it is important!
    _, share_net = get_network_byname(inputs=image_batch,
                                      config=net_config,
                                      is_training=IS_TRAINING,
                                      reuse=tf.AUTO_REUSE)
    # ***********************************************************************************************
    # *                                            FPN                                              *
    # ***********************************************************************************************
    feature_pyramid = build_fpn.build_feature_pyramid(share_net, net_config)
    # ***********************************************************************************************
    # *                                            RPN                                            *
    # ***********************************************************************************************
    rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                        image_window=image_window,
                        config=net_config)

    # rpn_proposals_scores==(2000,)
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(IS_TRAINING)

    # ***********************************************************************************************
    # *                                   Fast RCNN Head                                           *
    # ***********************************************************************************************

    fast_rcnn = build_head.FPNHead(feature_pyramid=feature_pyramid,
                                   rpn_proposals_boxes=rpn_proposals_boxes,
                                   origin_image=origin_image_batch,
                                   gtboxes_and_label=None,
                                   config=net_config,
                                   is_training=IS_TRAINING,
                                   image_window=image_window)

    detections = fast_rcnn.head_detection()
    
    # ***********************************************************************************************
    # *                                          Summary                                            *
    # ***********************************************************************************************

    if mode == tf.estimator.ModeKeys.PREDICT:
        predicts = {"image": origin_image_batch,
                    "predict_bbox": detections[:, :, :4],
                    "predict_class_id": detections[:, :, 4], "predict_scores": detections[:, :, 5],
                    "rpn_proposal_boxes": rpn_proposals_boxes,
                    "rpn_proposals_scores": rpn_proposals_scores,
                    "gt_box_labels": features["gt_box_labels"]}

        return tf.estimator.EstimatorSpec(mode, predictions=predicts)

