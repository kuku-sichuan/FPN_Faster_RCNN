# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import sys
sys.path.append('../')
from multiprocessing import cpu_count
import tensorflow as tf
from data import image_preprocess
from libs.box_utils import boxes_utils, make_anchor


def train_parse_fn(example, config):
    """
    :param example: 序列化的输入
    :return:
    """
    features = tf.parse_single_example(
        serialized=example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string)
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, shape=[img_height, img_width, 3])
    
    gt_boxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gt_boxes_and_label = tf.reshape(gt_boxes_and_label, [-1, 5])
    # shape of img is (1024, 1024, 3), image_window(4,)[y1, x1, y2, x2]
    img, gt_boxes_and_label, image_window = \
        image_preprocess.image_resize_pad(img_tensor=img,
                                          gtboxes_and_label=gt_boxes_and_label,
                                          target_side=config.TARGET_SIDE)
    img, gt_boxes_and_label = image_preprocess.random_flip_left_right(img_tensor=img,
                                                                      gtboxes_and_label=gt_boxes_and_label)
    # choose or padding make the gt_bbox_labels is FAST_RCNN_MAX_INSTANCES
    num_objects = tf.shape(gt_boxes_and_label)[0]
    object_index = tf.range(num_objects)
    object_index = tf.random_shuffle(object_index)
    object_index = object_index[:config.IMAGE_MAX_INSTANCES]
    gt_boxes_and_label = tf.gather(gt_boxes_and_label, object_index)
    anchor = make_anchor.generate_pyramid_anchors(config, name="input_data_pyramid_anchors")
    minibatch_indices, minibatch_encode_gtboxes, \
    rpn_objects_one_hot = boxes_utils.build_rpn_target(gt_boxes_and_label[:, :4], anchor, config)
    num_padding = config.IMAGE_MAX_INSTANCES - tf.shape(gt_boxes_and_label)[0]
    # (FAST_RCNN_MAX_INSTANCES, 5)[y1, x1, y2, x2, label]
    num_padding = tf.maximum(num_padding, 0)
    gt_box_label_padding = tf.zeros((num_padding, 5), dtype=tf.int32)
    gt_boxes_and_label = tf.concat([gt_boxes_and_label, gt_box_label_padding], axis=0)

    return {"image_name": img_name, "image": img, "image_window": image_window}, \
           {"gt_box_labels": gt_boxes_and_label, "minibatch_indices": minibatch_indices,
            "minibatch_encode_gtboxes": minibatch_encode_gtboxes,
            "minibatch_objects_one_hot": rpn_objects_one_hot}


def predict_parse_fn(example, config):

    features = tf.parse_single_example(
        serialized=example,
        features={
            'img_name': tf.FixedLenFeature([], tf.string),
            'img_height': tf.FixedLenFeature([], tf.int64),
            'img_width': tf.FixedLenFeature([], tf.int64),
            'img': tf.FixedLenFeature([], tf.string),
            'gtboxes_and_label': tf.FixedLenFeature([], tf.string),
        }
    )
    img_name = features['img_name']
    img_height = tf.cast(features['img_height'], tf.int32)
    img_width = tf.cast(features['img_width'], tf.int32)
    img = tf.decode_raw(features['img'], tf.uint8)
    img = tf.reshape(img, shape=[img_height, img_width, 3])

    # img.set_shape([None, None, 3])
    gt_boxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
    gt_boxes_and_label = tf.reshape(gt_boxes_and_label, [-1, 5])
    img, gt_boxes_and_label, image_window = \
        image_preprocess.image_resize_pad(img_tensor=img,
                                          gtboxes_and_label=gt_boxes_and_label,
                                          target_side=config.TARGET_SIDE)

    return {"image_name": img_name, "image": img,
            "image_window": image_window,
            "gt_box_labels": gt_boxes_and_label}


def train_input_fn(config):

    dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR, config.DATASET_NAME, config.TRAIN_DATASET_NAME))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
                            buffer_size=config.BATCH_SIZE * 4,
                            count=config.EPOCH))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda x: train_parse_fn(x, config),
                                                               config.BATCH_SIZE,
                                                               num_parallel_batches=cpu_count()//2))
    dataset = dataset.prefetch(config.BATCH_SIZE * 4)
    return dataset


def predict_input_fn(config):
    dataset = tf.data.TFRecordDataset(os.path.join(config.DATA_DIR, config.DATASET_NAME, config.EVAL_DATASET_NAME))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(
                                                      lambda x: predict_parse_fn(x, config),
                                                      1,
                                                      num_parallel_batches=cpu_count() // 2))
    return dataset



from config import TCTConfig

if __name__ == "__main__":
    net_config = TCTConfig()
    data = train_input_fn(net_config)
    dataset_iterator = data.make_one_shot_iterator()
    features_iter, labels_iter = dataset_iterator.get_next()
    with tf.Session() as sess:
        a,b = sess.run([features_iter, labels_iter])
        print(a)
        print(b)

