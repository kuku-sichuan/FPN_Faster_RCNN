# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from libs import losses
from libs.box_utils import encode_and_decode, boxes_utils
from libs.box_utils.show_box_in_tensor import draw_boxes_with_categories

layers = tf.keras.layers


class FPNHead(object):
    def __init__(self,
                 feature_pyramid,
                 rpn_proposals_boxes,
                 gtboxes_and_label,  # [batch_size, M, 5]
                 origin_image,
                 config,
                 is_training,
                 image_window):

        self.feature_pyramid = feature_pyramid
        self.rpn_proposals_boxes = rpn_proposals_boxes  # [batch_size, N, 4]
        self.gtboxes_and_label = gtboxes_and_label
        self.origin_image = origin_image
        self.config = config
        self.IS_TRAINING = is_training
        self.window = image_window
        self.level = config.LEVEL
        self.min_level = int(self.level[0][1])
        self.max_level = min(int(self.level[-1][1]), 5)

    def merge_batch_and_bboxes_dims(self, inputs):
        """
        :param inputs:list of tensor
        :return: list of tensor
        """
        outputs = []
        for input in inputs:
            input_shape = input.get_shape().as_list()
            output =tf.reshape(input, [-1,] + input_shape[2:])
            outputs.append(output)
        if len(inputs) == 1:
            return outputs[0]
        return outputs

    def div_batch_and_bboxes_dims(self, inputs):
        """
        :param inputs:list of tensor
        :return: list of tensor
        """
        outputs = []
        for input in inputs:
            input_shape = input.get_shape().as_list()
            output = tf.reshape(input, [self.config.PER_GPU_IMAGE, -1,] + input_shape[1:])
            outputs.append(output)
        if len(inputs) == 1:
            return outputs[0]
        return outputs

    def build_head_train_sample(self):

        """
        when training, we should know each reference box's label and gtbox,
        in second stage
        iou >= 0.5 is object
        iou < 0.5 is background
        this function need batch_slice
        :return:
        minibatch_reference_proboxes: (batch_szie, config.FAST_RCNN_MINIBATCH_SIZE, 4)[y1, x1, y2, x2]
        minibatch_encode_gtboxes:(batch_szie, config.FAST_RCNN_MINIBATCH_SIZE, 4)[dy, dx, log(dh), log(dw)]
        object_mask:(batch_szie, config.FAST_RCNN_MINIBATCH_SIZE) 1 indicate is object, 0 indicate is not objects
        label_one_hot: (batch_szie, config.FAST_RCNN_MINIBATCH_SIZE, num_class)
        """

        with tf.name_scope('build_head_train_sample'):

            def batch_slice_build_sample(gtboxes_and_label, rpn_proposals_boxes, config):

                with tf.name_scope('select_pos_neg_samples'):
                    gtboxes = tf.cast(
                        tf.reshape(gtboxes_and_label[:, :-1], [-1, 4]), tf.float32)
                    gt_class_ids = tf.cast(
                        tf.reshape(gtboxes_and_label[:, -1], [-1, ]), tf.int32)
                    gtboxes, non_zeros = boxes_utils.trim_zeros_graph(gtboxes, name="trim_gt_box")  # [M, 4]
                    gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)
                    rpn_proposals_boxes, _ = boxes_utils.trim_zeros_graph(rpn_proposals_boxes,
                                                                          name="trim_rpn_proposal_train")

                    ious = boxes_utils.iou_calculate(rpn_proposals_boxes, gtboxes)  # [N, M]
                    matchs = tf.cast(tf.argmax(ious, axis=1), tf.int32)  # [N, ]
                    max_iou_each_row = tf.reduce_max(ious, axis=1)
                    positives = tf.cast(tf.greater_equal(max_iou_each_row, config.FAST_RCNN_IOU_POSITIVE_THRESHOLD), tf.int32)
                    ignores = tf.cast(tf.less(max_iou_each_row, config.FAST_RCNN_IOU_LOW_NEG_THRESHOLD), tf.int32)* -1
                    positives = positives + ignores

                    reference_boxes_mattached_gtboxes = tf.gather(gtboxes, matchs)  # [N, 4]
                    gt_class_ids = tf.gather(gt_class_ids, matchs)  # [N, ]
                    object_mask = tf.cast(positives, tf.float32)  # [N, ]
                    # when box is background, not caculate gradient, so give a weight 0 to avoid caculate gradient
                    gt_class_ids = gt_class_ids * positives

                with tf.name_scope('head_train_minibatch'):
                    # choose the positive indices
                    positive_indices = tf.reshape(tf.where(tf.equal(object_mask, 1.)), [-1])
                    num_of_positives = tf.minimum(tf.shape(positive_indices)[0],
                                                  tf.cast(config.FAST_RCNN_MINIBATCH_SIZE*config.FAST_RCNN_POSITIVE_RATE,
                                                          tf.int32))
                    positive_indices = tf.random_shuffle(positive_indices)
                    positive_indices = tf.slice(positive_indices, begin=[0], size=[num_of_positives])
                    # choose the negative indices,
                    # Strictly propose the proportion of positive and negative is 1:3
                    negative_indices = tf.reshape(tf.where(tf.equal(object_mask, 0.)), [-1])
                    num_of_negatives = tf.cast(int(1. / config.FAST_RCNN_POSITIVE_RATE) * num_of_positives, tf.int32)\
                                       - num_of_positives

                    num_of_negatives = tf.minimum(tf.shape(negative_indices)[0], num_of_negatives)
                    negative_indices = tf.random_shuffle(negative_indices)
                    negative_indices = tf.slice(negative_indices, begin=[0], size=[num_of_negatives])

                    minibatch_indices = tf.concat([positive_indices, negative_indices], axis=0)
                    minibatch_reference_gtboxes = tf.gather(reference_boxes_mattached_gtboxes,
                                                            minibatch_indices)
                    minibatch_reference_proboxes = tf.gather(rpn_proposals_boxes, minibatch_indices)
                    # encode gtboxes
                    minibatch_encode_gtboxes = \
                        encode_and_decode.encode_boxes(
                            unencode_boxes=minibatch_reference_gtboxes,
                            reference_boxes=minibatch_reference_proboxes,
                            dev_factors=config.BBOX_STD_DEV)
                    object_mask = tf.gather(object_mask, minibatch_indices)
                    gt_class_ids = tf.gather(gt_class_ids, minibatch_indices)

                    # padding if necessary
                    gap = tf.cast(config.FAST_RCNN_MINIBATCH_SIZE - (num_of_positives + num_of_negatives), dtype=tf.int32)
                    bbox_padding = tf.zeros((gap, 4))
                    minibatch_reference_proboxes = tf.concat([minibatch_reference_proboxes, bbox_padding], axis=0)
                    minibatch_encode_gtboxes = tf.concat([minibatch_encode_gtboxes, bbox_padding], axis=0)
                    object_mask = tf.pad(object_mask, [(0, gap)])
                    gt_class_ids = tf.pad(gt_class_ids, [(0, gap)])

                return minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids

            minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids = \
                    boxes_utils.batch_slice([self.gtboxes_and_label, self.rpn_proposals_boxes],
                                            lambda x, y: batch_slice_build_sample(x, y, self.config),
                                            self.config.PER_GPU_IMAGE)
        if self.config.DEBUG:
            gt_vision = draw_boxes_with_categories(self.origin_image[0],
                                                   self.gtboxes_and_label[0, :, :4],
                                                   self.gtboxes_and_label[0, :, 4])
            tf.summary.image("gt_vision", gt_vision)

            draw_bbox_train = draw_boxes_with_categories(self.origin_image[0],
                                                         minibatch_reference_proboxes[0],
                                                         gt_class_ids[0])
            tf.summary.image("train_proposal", draw_bbox_train)

        return minibatch_reference_proboxes, minibatch_encode_gtboxes, object_mask, gt_class_ids

    def assign_level(self, minibatch_reference_proboxes):

        """
        compute the level of rpn_proposals_boxes
        :param: minibatch_reference_proboxes (batch_size, num_proposals, 4)[y1, x1, y2, x2]
        return: (batch_size, num_proposals)
        Note that we have not trim the elements padding is 0 which does not affect the finial result.
        """
        with tf.name_scope('assign_levels'):
            ymin, xmin, ymax, xmax = tf.unstack(minibatch_reference_proboxes, axis=2)

            w = tf.maximum(xmax - xmin, 0.)  # avoid w is negative
            h = tf.maximum(ymax - ymin, 0.)  # avoid h is negative

            levels = tf.round(4. + tf.log(tf.sqrt(w*h + 1e-8)/224.0) / tf.log(2.))  # 4 + log_2(***)

            levels = tf.maximum(levels, tf.ones_like(levels) * (np.float32(self.min_level)))  # level minimum is 2
            levels = tf.minimum(levels, tf.ones_like(levels) * (np.float32(self.max_level)))  # level maximum is 5

            return tf.cast(levels, tf.int32)

    def get_rois_feature(self, proposal_bbox):

        """
        1)get roi from feature map
        2)roi align or roi pooling. Here is roi align
        :param: proposal_bbox: (batch_size, num_proposal, 4)[y1, x1, y2, x2]
        :return:
        all_level_rois: [batch_size, num_proposal, 7, 7, C]
        """

        levels = self.assign_level(proposal_bbox)
        with tf.name_scope('obtain_roi_feature'):
            pooled = []
            # this is aimed at reorder the pooling map (batch_size, num_proposal)
            box_to_level = []
            for i in range(self.min_level, self.max_level + 1):
                ix = tf.where(tf.equal(levels, i))
                level_i_proposals = tf.gather_nd(proposal_bbox, ix)

                # Box indicies for crop_and_resize.
                box_indices = tf.cast(ix[:, 0], tf.int32)

                box_to_level.append(ix)

                level_i_proposals = tf.stop_gradient(level_i_proposals)
                box_indices = tf.stop_gradient(box_indices)

                image_shape = tf.constant([self.config.TARGET_SIDE-1, self.config.TARGET_SIDE-1,
                                           self.config.TARGET_SIDE-1, self.config.TARGET_SIDE-1], dtype=tf.float32)
                normal_level_i_proposals = level_i_proposals / image_shape

                level_i_cropped_rois = tf.image.crop_and_resize(self.feature_pyramid['P%d' % i],
                                                                boxes=normal_level_i_proposals,
                                                                box_ind=box_indices,
                                                                crop_size=[self.config.ROI_SIZE, self.config.ROI_SIZE])
                pooled.append(level_i_cropped_rois)

            # Pack pooled features into one tensor
            pooled = tf.concat(pooled, axis=0)
            box_to_level = tf.concat(box_to_level, axis=0)
            box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
            box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                     axis=1)

            # Rearrange pooled features to match the order of the original boxes
            # Sort box_to_level by batch then box index
            # TF doesn't have a way to sort by two columns, so merge them and sort.
            sorting_tensor = box_to_level[:, 0] * 10000 + box_to_level[:, 1]
            ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
                box_to_level)[0]).indices[::-1]
            ix = tf.gather(box_to_level[:, 2], ix)
            pooled = tf.gather(pooled, ix)
            reshape_pooled = self.div_batch_and_bboxes_dims([pooled])
            return reshape_pooled

    def head_net(self, features, is_training):
        """
        base the feature to compute the finial bbox and scores
        :param features:(batch_size, num_proposal, 7, 7, channels)
        :param is_training: whether change the mean and variance in batch_normalization
        :return:
        fast_rcnn_encode_boxes: (batch_size, num_proposal, num_classes*4)
        fast_rcnn_scores:(batch_size, num_proposal, num_classes)
        """

        def batch_slice_head_net(features, config, is_training):

            with tf.variable_scope('head_net', reuse=tf.AUTO_REUSE):

                net = layers.Conv2D(filters=1024,
                                    kernel_size=(self.config.ROI_SIZE, config.ROI_SIZE),
                                    name="fc1")(features)
                net = tf.layers.batch_normalization (net, training=is_training)
                net = layers.Activation('relu6')(net)
                net = layers.Conv2D(filters=1024,
                                    kernel_size=(1, 1),
                                    name="fc2")(inputs=net)
                net = tf.layers.batch_normalization (net, training=is_training)
                net = layers.Activation('relu6')(net)

                net = tf.squeeze(net, axis=[1, 2])
                head_scores = layers.Dense(config.NUM_CLASS,
                                           name='head_classifier')(net)
                head_encode_boxes = layers.Dense(net, config.NUM_CLASS * 4,
                                                 name='head_regressor')
                head_encode_boxes = layers.Reshape((config.NUM_CLASS, 4))(head_encode_boxes)

                return head_encode_boxes, head_scores

        head_encode_boxes, head_scores =\
            boxes_utils.batch_slice([features],
                                    lambda x: batch_slice_head_net(x, self.config, is_training),
                                    self.config.PER_GPU_IMAGE)

        return head_encode_boxes, head_scores

    def head_loss(self):

        minibatch_reference_proboxes, minibatch_encode_gtboxes,\
        object_mask, gt_class_ids = self.build_head_train_sample

        pooled_feature = self.get_rois_feature(minibatch_reference_proboxes)

        fast_rcnn_predict_boxes, fast_rcnn_predict_scores = self.head_net(pooled_feature, self.IS_TRAINING)

        with tf.variable_scope("head_loss"):
            # trim zero graph
            # minibatch_encode_gtboxes, non_zeros = boxes_utils.trim_zeros_graph(minibatch_encode_gtboxes,
            #                                                                    name="trim_gtbox_finial_loss")
            # object_mask = tf.boolean_mask(object_mask, non_zeros)
            # fast_rcnn_predict_boxes = tf.boolean_mask(fast_rcnn_predict_boxes, non_zeros)
            # fast_rcnn_predict_boxes = tf.reshape(fast_rcnn_predict_boxes, [-1, self.config.NUM_CLASS, 4])
            #
            # fast_rcnn_predict_scores = tf.boolean_mask(fast_rcnn_predict_scores, non_zeros)
            # label_one_hot = tf.boolean_mask(label_one_hot, non_zeros)

            # from fast_rcnn_predict_boxes choose corresponding encode
            row_index = tf.range(0, tf.shape(gt_class_ids)[1])
            row_index = tf.expand_dims(row_index, 0)
            multi_row_index = tf.tile(row_index, [self.config.PER_GPU_IMAGE, 1])
            multi_row_index = tf.expand_dims(multi_row_index, axis=-1)
            expand_gt_class_ids = tf.expand_dims(gt_class_ids, axis=-1)
            index = tf.concat([multi_row_index, expand_gt_class_ids], axis=-1)
            fast_rcnn_predict_boxes = boxes_utils.batch_slice([fast_rcnn_predict_boxes, index],
                                                              lambda x, y: tf.gather_nd(x, y),
                                                              self.config.PER_GPU_IMAGE)

            # loss
            with tf.variable_scope('head_class_loss'):
                fast_rcnn_classification_loss = tf.losses.sparse_softmax_cross_entropy(labels=gt_class_ids,
                                                                                       logits=fast_rcnn_predict_scores)

                fast_rcnn_classification_loss = tf.cond(tf.is_nan(fast_rcnn_classification_loss), lambda: 0.0,
                                                        lambda: fast_rcnn_classification_loss)

            with tf.variable_scope('head_location_loss'):
                fast_rcnn_location_loss = losses.l1_smooth_losses(predict_boxes=fast_rcnn_predict_boxes,
                                                                  gtboxes=minibatch_encode_gtboxes,
                                                                  object_weights=object_mask)

            return fast_rcnn_location_loss, fast_rcnn_classification_loss

    def head_proposals(self, rpn_proposal_bbox, encode_boxes, categories, scores, image_window):
        """
        padding zeros to keep alignments
        :return:
        detection_boxes_scores_labels:(batch_size, config.MAX_DETECTION_INSTANCE, 6)
        """

        def batch_slice_head_proposals(rpn_proposal_bbox,
                                       encode_boxes,
                                       categories,
                                       scores,
                                       image_window,
                                       config):
            """
            mutilclass NMS
            :param rpn_proposal_bbox: (N, 4)
            :param encode_boxes: (N, 4)
            :param categories:(N, )
            :param scores: (N, )
            :param image_window:(y1, x1, y2, x2) the boundary of image
            :return:
            detection_boxes_scores_labels : (-1, 6)[y1, x1, y2, x2, scores, labels]
            """
            with tf.name_scope('head_proposals'):
                # trim the zero graph
                rpn_proposal_bbox, non_zeros = boxes_utils.trim_zeros_graph(rpn_proposal_bbox,
                                                                            name="trim_proposals_detection")
                encode_boxes = tf.boolean_mask(encode_boxes, non_zeros)
                categories = tf.boolean_mask(categories, non_zeros)
                scores = tf.boolean_mask(scores, non_zeros)
                fast_rcnn_decode_boxes = encode_and_decode.decode_boxes(encode_boxes=encode_boxes,
                                                                        reference_boxes=rpn_proposal_bbox,
                                                                        dev_factors=config.BBOX_STD_DEV)
                fast_rcnn_decode_boxes = boxes_utils.clip_boxes_to_img_boundaries(fast_rcnn_decode_boxes,
                                                                                  image_window)

                # remove the background
                keep = tf.cast(tf.where(categories > 0)[:, 0], tf.int32)
                if self.config.DEBUG:
                    print_categories = tf.gather(categories, keep)
                    print_scores = tf.gather(scores, keep)
                    num_item = tf.minimum(tf.shape(print_scores)[0], 50)
                    print_scores_vision, print_index = tf.nn.top_k(print_scores, k=num_item)
                    print_categories_vision = tf.gather(print_categories, print_index)
                    boxes_utils.print_tensors(print_categories_vision, "categories")
                    boxes_utils.print_tensors(print_scores_vision, "scores")
                # Filter out low confidence boxes
                if config.FINAL_SCORE_THRESHOLD:
                    conf_keep = tf.cast(tf.where(scores >= config.FINAL_SCORE_THRESHOLD)[:, 0], tf.int32)
                    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                                    tf.expand_dims(conf_keep, 0))
                    keep = tf.sparse_tensor_to_dense(keep)[0]

                pre_nms_class_ids = tf.gather(categories, keep)
                pre_nms_scores = tf.gather(scores, keep)
                pre_nms_rois = tf.gather(fast_rcnn_decode_boxes, keep)
                unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

                def nms_keep_map(class_id):
                    """Apply Non-Maximum Suppression on ROIs of the given class."""
                    # Indices of ROIs of the given class
                    ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
                    # Apply NMS
                    class_keep = tf.image.non_max_suppression(
                        tf.gather(pre_nms_rois, ixs),
                        tf.gather(pre_nms_scores, ixs),
                        max_output_size=config.DETECTION_MAX_INSTANCES,
                        iou_threshold=config.FAST_RCNN_NMS_IOU_THRESHOLD)
                    # Map indicies
                    class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
                    # Pad with -1 so returned tensors have the same shape
                    gap = config.DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
                    class_keep = tf.pad(class_keep, [(0, gap)],
                                        mode='CONSTANT', constant_values=-1)
                    # Set shape so map_fn() can infer result shape
                    class_keep.set_shape([config.DETECTION_MAX_INSTANCES])
                    return class_keep
                # 2. Map over class IDs
                nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                                     dtype=tf.int32)
                # 3. Merge results into one list, and remove -1 padding
                nms_keep = tf.reshape(nms_keep, [-1])
                nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
                # 4. Compute intersection between keep and nms_keep
                keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                                tf.expand_dims(nms_keep, 0))
                keep = tf.sparse_tensor_to_dense(keep)[0]
                # Keep top detections
                roi_count = config.DETECTION_MAX_INSTANCES
                class_scores_keep = tf.gather(scores, keep)
                num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
                top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
                keep = tf.gather(keep, top_ids)

                # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
                # Coordinates are normalized.
                detections = tf.concat([
                    tf.gather(fast_rcnn_decode_boxes, keep),
                    tf.to_float(tf.gather(categories, keep))[..., tf.newaxis],
                    tf.gather(scores, keep)[..., tf.newaxis]
                ], axis=1)

                # Pad with zeros if detections < DETECTION_MAX_INSTANCES
                gap = config.DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
                detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")

                return detections

        detections = boxes_utils.batch_slice([rpn_proposal_bbox, encode_boxes, categories, scores, image_window],
                                             lambda x, y, z, u, v: batch_slice_head_proposals(x, y, z, u, v,
                                                                                              self.config),
                                             self.config.PER_GPU_IMAGE)
        return detections

    def head_detection(self):
        """
        compute the predict bboxes, categories, categories
        :return:
        fast_rcnn_categories_bboxs:(batch_size, num_proposals, 4)
        fast_rcnn_categories_scores:(batch_size, num_propsals)
        fast_rcnn_categories:(batch_size, num_propsals)
        """


        # (batch_size, num_proposal, 7, 7, channels)
        pooled_feature = self.get_rois_feature(self.rpn_proposals_boxes)
        head_predict_boxes, head_predict_scores = self.head_net(pooled_feature, False)

        with tf.name_scope("head_detection"):

            head_softmax_scores = layers.Softmax()(head_predict_scores)  # [N, -1, num_classes]
            # gain the highest category and score and bounding box
            head_categories = tf.argmax(head_softmax_scores, axis=2, output_type=tf.int32) # (N, -1)
            row_index = tf.range(0, tf.shape(head_categories)[1])
            row_index = tf.expand_dims(row_index, 0)
            multi_row_index = tf.tile(row_index, [self.config.PER_GPU_IMAGE, 1])
            multi_row_index = tf.expand_dims(multi_row_index, axis=-1)
            expand_head_categories = tf.expand_dims(head_categories, axis=-1)
            index = tf.concat([multi_row_index, expand_head_categories], axis=-1)
            head_categories_bboxs = boxes_utils.batch_slice([head_predict_boxes, index],
                                                                 lambda x, y: tf.gather_nd(x, y),
                                                                 self.config.PER_GPU_IMAGE)
            head_categories_scores = tf.reduce_max(head_softmax_scores, axis=2, keepdims=False)# (N, -1)

            detections = self.head_proposals(self.rpn_proposals_boxes,
                                             head_categories_bboxs,
                                             head_categories,
                                             head_categories_scores,
                                             self.window)

            return detections












