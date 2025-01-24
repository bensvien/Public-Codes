# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:04:55 2025

@author: Dr. Ben Vien
"""

import tensorflow as tf

class PINN(tf.keras.Model):
    def __init__(self, layers):
        super(PINN, self).__init__()
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='tanh') for units in layers[:-1]]
        self.output_layer = tf.keras.layers.Dense(layers[-1])

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


#%%
model = PINN([2, 64, 64, 64, 1])  # 2,64,64,64,1 dense Sample-test
