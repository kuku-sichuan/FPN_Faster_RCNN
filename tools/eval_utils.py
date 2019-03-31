
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from libs.box_utils.boxes_utils import iou_calculate, trim_zeros_graph


def compute_metric_ap(gt_boxes,
                      gt_class_ids,
                      pred_boxes,
                      pred_class_ids,
                      pred_scores,
                      config,
                      iou_threshold=0.5):

    """Compute Matching status at a set IoU threshold (default 0.5).

        Returns:
        match_gt_label:{1:Tensor(M1,),2:Tensor(M2,)...}
        match_pred_label:{1:Tensor(M1,),2:Tensor(M2,)...}
     """
    gt_boxes, gt_non_zeros = trim_zeros_graph(gt_boxes)
    gt_class_ids = tf.boolean_mask(gt_class_ids, gt_non_zeros)
    pred_boxes, pred_non_zeros = trim_zeros_graph(pred_boxes)
    pred_class_ids = tf.boolean_mask(pred_class_ids, pred_non_zeros)
    pred_scores = tf.boolean_mask(pred_scores, pred_non_zeros)

    sorted_index = tf.contrib.framework.argsort(pred_scores,
                                                axis=-1,
                                                direction='DESCENDING')
    pred_boxes = tf.gather(pred_boxes, sorted_index)
    pred_scores = tf.gather(pred_scores, sorted_index)
    pred_class_ids = tf.gather(pred_class_ids, sorted_index)

    pred_gt_ious = iou_calculate(pred_boxes, gt_boxes)

    # build the matrix which means the iou between gt and pred is more than 0.5
    greater_iou_matrix = tf.greater_equal(pred_gt_ious, iou_threshold)
    # build the matrix which means the label between gt and pred is equal.
    label_equal_matrix = tf.equal(tf.expand_dims(pred_class_ids, axis=1),
                                  tf.expand_dims(gt_class_ids, axis=0))
    # obtain the location which has same label and iou is bigger than iou_thresh
    match_matrix = tf.logical_and(greater_iou_matrix, label_equal_matrix)
    match_matrix_int = tf.cast(match_matrix, tf.int32)
    match_iou = tf.multiply(pred_gt_ious, match_matrix_int)
    # Remove duplicate elements in a row
    single_match_pred = tf.where(tf.logical_and(tf.equal(match_iou,
                                       tf.reduce_max(match_iou, axis=1,
                                                     keep_dims=True),
                              tf.greater_equal(match_iou, iou_threshold))),
                              1, 0)
    # Remove duplicate elements in a column
    first_one_element = tf.one_hot(tf.argmax(single_match_pred, axis=0),
                                   depth=tf.shape(single_match_pred)[0],
                                   axis=0)
    single_match = tf.multiply(single_match_pred, first_one_element)

    match_iou = tf.multiply(single_match, match_iou)
    related_gt_box = tf.argmax(single_match, axis=-1)
    match_pred_gt_label = tf.gather(gt_class_ids, related_gt_box)

    # to set some pred to 0 because it's FP or double
    max_iou = tf.reduce_max(match_iou, axis=-1)
    iou_bigger_threshold = tf.greater_equal(max_iou, iou_threshold)
    iou_bigger_thres_int = tf.cast(iou_bigger_threshold, tf.int32)
    pred_gt_label = tf.multiply(match_pred_gt_label, iou_bigger_thres_int)

    # add some instance which missed
    miss_gt = tf.where(tf.equal(tf.reduce_sum(single_match, axis=0), 0))
    miss_gt_label = tf.gather(gt_class_ids, miss_gt)
    correspond_pred = tf.zeros_like(miss_gt_label)

    # concat the missed and prediction
    eval_pred_class_ids = tf.concat(pred_class_ids, correspond_pred, axis=0)
    eval_gt_class_ids = tf.concat(gt_class_ids, miss_gt_label, axis=0)

    eval_metrics = {}
    for i in range(1, config.NUM_CLASS):
        temp_index = tf.where(tf.logical_or(
                              tf.equal(eval_pred_class_ids, i),
                              tf.equal(eval_gt_class_ids, i)))
        temp_pred = tf.gather(eval_pred_class_ids, temp_index)
        temp_gt = tf.gather(eval_gt_class_ids, temp_index)
        eval_metrics[config.LABEL_TO_NAME[i]] = \
            tf.metrics.average_precision_at_k(temp_gt,
                                              tf.one_hot(temp_pred,
                                                         depth=config.NUM_CLASS, axis=1),
                                              1)
        return eval_metrics








