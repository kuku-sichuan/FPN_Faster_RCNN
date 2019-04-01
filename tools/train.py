# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
sys.path.append('../')
import os
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from libs.networks.network_factory import get_network_byname
from libs import build_rpn, build_head, build_fpn
from libs.box_utils.show_box_in_tensor import draw_boxes_with_scores, draw_boxes_with_categories_and_scores
from libs.box_utils.boxes_utils import batch_slice
from data.read_tfrecord import train_input_fn
from config import TCTConfig
from tools.run_meta import MetadataHook
from tools.eval_utils import compute_metric_ap
from libs.box_utils.boxes_utils import print_tensors

tf.logging.set_verbosity(tf.logging.INFO)


def model_fn(features,
             labels,
             mode,
             params):

    # ***********************************************************************************************
    # *                                         Backbone Net                                           *
    # ***********************************************************************************************
    net_config = params["net_config"]
    if mode == tf.estimator.ModeKeys.TRAIN:
        IS_TRAINING = True
    else:
        IS_TRAINING = False

    origin_image_batch = features["image"]
    image_batch = origin_image_batch - tf.convert_to_tensor(net_config.PIXEL_MEANS, dtype=tf.float32)
    image_window = features["image_window"]
    # there is is_training means that bn is training, so it is important!
    _, share_net = get_network_byname(inputs=image_batch,
                                      config=net_config,
                                      is_training=False,
                                      reuse=tf.AUTO_REUSE)
    # ***********************************************************************************************
    # *                                      FPN                                          *
    # ***********************************************************************************************
    feature_pyramid = build_fpn.build_feature_pyramid(share_net, net_config)
    # ***********************************************************************************************
    # *                                      RPN                                             *
    # ***********************************************************************************************
    gtboxes_and_label_batch = labels.get("gt_box_labels")
    rpn = build_rpn.RPN(feature_pyramid=feature_pyramid,
                        image_window=image_window,
                        config=net_config)

    # rpn_proposals_scores==(2000,)
    rpn_proposals_boxes, rpn_proposals_scores = rpn.rpn_proposals(IS_TRAINING)
    rpn_location_loss, rpn_classification_loss = rpn.rpn_losses(labels["minibatch_indices"],
                                                                labels["minibatch_encode_gtboxes"],
                                                                labels["minibatch_objects_one_hot"])
        
    rpn_total_loss = rpn_classification_loss + rpn_location_loss

    # ***********************************************************************************************
    # *                                   Fast RCNN Head                                          *
    # ***********************************************************************************************

    fpn_fast_rcnn_head = build_head.FPNHead(feature_pyramid=feature_pyramid,
                                            rpn_proposals_boxes=rpn_proposals_boxes,
                                            origin_image=origin_image_batch,
                                            gtboxes_and_label=gtboxes_and_label_batch,
                                            config=net_config,
                                            is_training=False,
                                            image_window=image_window)

    detections = fpn_fast_rcnn_head.head_detection()
    if net_config.DEBUG:
        print_tensors(rpn_proposals_scores[0,:50],"scores")
        print_tensors(rpn_proposals_boxes[0, :50, :], "bbox")
        rpn_proposals_vision = draw_boxes_with_scores(origin_image_batch[0, :, :, :],
                                                      rpn_proposals_boxes[0, :50, :],
                                                      rpn_proposals_scores[0, :50])
        head_vision = draw_boxes_with_categories_and_scores(origin_image_batch[0, :, :, :],
                                                            detections[0, :, :4],
                                                            detections[0, :, 4],
                                                            detections[0, :, 5],
                                                            net_config.LABEL_TO_NAME)
        tf.summary.image("rpn_proposals_vision", rpn_proposals_vision) 
        tf.summary.image("head_vision", head_vision)
    
    head_location_loss, head_classification_loss = fpn_fast_rcnn_head.head_loss()
    head_total_loss = head_location_loss + head_classification_loss

    # train
    with tf.name_scope("regularization_losses"):
        regularization_list = [tf.nn.l2_loss(w.read_value()) *
                               net_config.WEIGHT_DECAY / tf.cast(tf.size(w.read_value()),
                               tf.float32) for w in tf.trainable_variables() if 'gamma' not
                               in w.name and 'beta' not in w.name]
        regularization_loss = tf.add_n(regularization_list)

    total_loss = regularization_loss + head_total_loss + rpn_total_loss
    total_loss = tf.cond(tf.is_nan(total_loss),lambda:0.0,lambda:total_loss)
    print_tensors(head_total_loss,"head_loss")
    print_tensors(rpn_total_loss,"rpn_loss")
    global_step = tf.train.get_or_create_global_step()
    tf.train.init_from_checkpoint(net_config.CHECKPOINT_DIR,
                                  {net_config.BACKBONE_NET + "/": net_config.BACKBONE_NET + "/"})
    with tf.name_scope("optimizer"):
        lr = tf.train.piecewise_constant(global_step,
                                         boundaries=[np.int64(net_config.BOUNDARY[0])],
                                         values=[net_config.LEARNING_RATE, net_config.LEARNING_RATE / 10])
        optimizer = tf.train.MomentumOptimizer(lr, momentum=net_config.MOMENTUM)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies([tf.group(*update_ops)]):
            grads = optimizer.compute_gradients(total_loss)
            # clip gradients
            grads = tf.contrib.training.clip_gradient_norms(grads, net_config.CLIP_GRADIENT_NORM)
            train_op = optimizer.apply_gradients(grads, global_step)

    # ***********************************************************************************************
    # *                                          Summary                                            *
    # ***********************************************************************************************
    # rpn loss and image
    tf.summary.scalar('rpn_location_loss', rpn_location_loss, family="rpn_loss")
    tf.summary.scalar('rpn_classification_loss', rpn_classification_loss, family="rpn_loss")
    tf.summary.scalar('rpn_total_loss', rpn_total_loss, family="rpn_loss")

    tf.summary.scalar('head_location_loss', head_location_loss, family="head_loss")
    tf.summary.scalar('head_classification_loss', head_classification_loss, family="head_loss")
    tf.summary.scalar('head_total_loss', head_total_loss, family="head_loss")
    tf.summary.scalar("regularization_loss", regularization_loss)
    tf.summary.scalar('total_loss', total_loss)
    tf.summary.scalar('learning_rate', lr)

    meta_hook = MetadataHook(save_steps=net_config.SAVE_EVERY_N_STEP * net_config.EPOCH / 2,
                             output_dir=net_config.MODLE_DIR)
    summary_hook = tf.train.SummarySaverHook(save_steps=net_config.SAVE_EVERY_N_STEP,
                                             output_dir=net_config.MODLE_DIR,
                                             summary_op=tf.summary.merge_all())
    hooks = [summary_hook]
    if net_config.COMPUTE_TIME:
        hooks.append(meta_hook)
    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                          train_op=train_op,
                                          training_hooks=hooks)

    # ***********************************************************************************************
    # *                                            EVAL                                             *
    # ***********************************************************************************************
    metric_ap_dict = batch_slice([features["gt_box_labels"][:,:,:4],
                                 features["gt_box_labels"][:, :, 4],
                                 detections[:, :, :4],
                                 detections[:, :, 4],
                                 detections[:, :, 5]],
                                 lambda x, y, z, u, v: compute_metric_ap(x, y, z, u, v, net_config),
                                 net_config.PER_GPU_IMAGE)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                          eval_metric_ops=metric_ap_dict)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    net_config = TCTConfig()
    session_config = tf.ConfigProto()
    session_config.gpu_options.allow_growth = True
    session_config.allow_soft_placement = True
    estimator_config = tf.estimator.RunConfig(model_dir=os.path.join(net_config.MODLE_DIR, net_config.NET_NAME),
                                              log_step_count_steps=200,
                                              save_summary_steps=net_config.SAVE_EVERY_N_STEP,
                                             save_checkpoints_steps=net_config.SAVE_EVERY_N_STEP,
                                              session_config=session_config)
    my_estimator = tf.estimator.Estimator(tf.contrib.estimator.replicate_model_fn(model_fn,
                                          devices=net_config.GPU_GROUPS),
                                          params={"net_config": net_config}, 
                                          config=estimator_config)
    my_estimator.train(input_fn=lambda: train_input_fn(net_config))

