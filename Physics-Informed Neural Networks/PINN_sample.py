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

def wave_equation_loss(model, X, c):
    with tf.GradientTape(persistent=True) as tape2:
        tape2.watch(X)
        with tf.GradientTape() as tape1:
            tape1.watch(X)
            u = model(X)
        u_gradients = tape1.gradient(u, X)  # Compute first derivatives
        u_x = u_gradients[:, 0:1]
        u_t = u_gradients[:, 1:2]
    u_xx = tape2.gradient(u_x, X)[:, 0:1]  # Second derivative w.r.t. x
    u_tt = tape2.gradient(u_t, X)[:, 1:2]  # Second derivative w.r.t. t
    del tape1, tape2  # Free up memory
    f = u_tt - c**2 * u_xx  # Wave equation residual
    return tf.reduce_mean(tf.square(f))

optimizer = tf.keras.optimizers.Adam()
#%%
model = PINN([2, 64, 64, 64, 1])  # 2,64,64,64,1 dense Sample-test
