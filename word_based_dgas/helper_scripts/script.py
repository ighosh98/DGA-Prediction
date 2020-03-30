#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 15:50:30 2020

@author: ighosh
"""

import tensorflow as tf
if tf.test.is_gpu_available():
    print tf.test.gpu_device_name()