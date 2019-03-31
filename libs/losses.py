# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def l1_smooth_losses(predict_boxes, gtboxes, object_weights):
    """
    :param predict_boxes: [batch_size, num_boxes, 4]
    :param gtboxes: [batch_size, num_boxes]

    :return:
    """
    # enhanced robustness
    gtboxes = tf.where(tf.is_nan(gtboxes), predict_boxes, gtboxes)
    # choose positive objects

    object_weights = tf.cast(object_weights, tf.int32)
    index = tf.cast(tf.where(tf.equal(object_weights, 1)), tf.int32)
    predict_boxes = tf.gather_nd(predict_boxes, index)
    gtboxes = tf.gather_nd(gtboxes, index)
    diff = predict_boxes - gtboxes
    abs_diff = tf.cast(tf.abs(diff), tf.float32)

    # avoid proposal is no objects
    smooth_box_loss = tf.cond(tf.size(gtboxes) > 0,
                              lambda: tf.reduce_mean(tf.where(tf.less(abs_diff, 1), 0.5 * tf.square(abs_diff), abs_diff - 0.5)),
                              lambda: 0.0)

    return smooth_box_loss


def my_sigmoid_cross_entropy(labels, predictions, num_class, smooth=False):
    """
    :param labels:(N,)
    :param predictions:(N, num_classes)
    
    """
    labels = tf.one_hot(labels, num_class)
    labels = tf.cond(smooth, true_fn=lambda:labels *(1-0.2) + 0.2/num_class,
                             false_fn=lambda: labels)
    return tf.losses.sigmoid_cross_entropy(labels, predictions)


def my_softmax_cross_entropy(labels, predictions, num_class, smooth=False):
    labels = tf.one_hot(labels, num_class)
    labels = tf.cond(smooth, true_fn=lambda:labels *(1-0.2) + 0.2/num_class,
                             false_fn=lambda: labels)
    return tf.losses.softmax_cross_entropy(labels, predictions)


def test_smoothl1():

    predict_boxes = tf.constant([[1, 1, 2, 2],
                                [2, 2, 2, 2],
                                [3, 3, 3, 3]])
    gtboxes = tf.constant([[1, 1, 1, 1],
                          [2, 1, 1, 1],
                          [3, 3, 2, 1]])

    loss = l1_smooth_losses(predict_boxes, gtboxes, [1, 1, 1])

    with tf.Session() as sess:
        print(sess.run(loss))

if __name__ == '__main__':
    test_smoothl1()
