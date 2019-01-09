# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 20:46:47 2019

@author: mail
"""

from __future__ import print_function
import tensorflow as tf
#导入GBDT分类器
from tensorflow.contrib.boosted_trees.estimator_batch.estimator import GradientBoostedDecisionTreeClassifier
from tensorflow.contrib.boosted_trees.proto import learner_pb2 as gbdt_learner

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '' #不用GPU

tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/',one_hot = False,
                                   source_url = 'http://yann.lecun.com/exdb/mnist/')

#参数
batch_size = 4096      #batch的大小
num_classes = 10        #样本的类别数
num_features = 784      #样本的特征数
max_steps = 1000         #最大迭代步数
#GBDT参数
learning_rate = 0.1
l1_regul = 0.
l2_regul = 1.
examples_per_layer = 1000
num_trees  = 10
max_depth = 16

learner_config = gbdt_learner.LearnerConfig()
learner_config.learning_rate_tuner.fixed.learning_rate = learning_rate
learner_config.regularization.l1 = l1_regul
learner_config.regularization.l2 = l2_regul / examples_per_layer
learner_config.constraints.max_tree_depth = max_depth
growing_mode = gbdt_learner.LearnerConfig.LAYER_BY_LAYER
learner_config.growing_mode = growing_mode

run_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=300)
learner_config.multi_class_strategy = (gbdt_learner.LearnerConfig.DIAGONAL_HESSIAN)\

#创建GDBT分类器
gbdt_model = GradientBoostedDecisionTreeClassifier(
        model_dir=None, # No save directory specified
        learner_config=learner_config,
        n_classes=num_classes,
        examples_per_layer=examples_per_layer,
        num_trees=num_trees,
        center_bias=False,
        config=run_config)

tf.logging.set_verbosity(tf.logging.INFO)
#输入
input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.train.images}, y=mnist.train.labels,
    batch_size=batch_size, num_epochs=None, shuffle=True)
# Train the Model
gbdt_model.fit(input_fn=input_fn, max_steps=max_steps)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'images': mnist.test.images}, y=mnist.test.labels,
    batch_size=batch_size, shuffle=False)
# Use the Estimator 'evaluate' method
e = gbdt_model.evaluate(input_fn=input_fn)

print("Testing Accuracy:", e['accuracy'])