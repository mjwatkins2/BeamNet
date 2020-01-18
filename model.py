from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import time

from layers import SpatialPowers

class BeamModel:
    """Provides for training a simple tensorflow network to represent a beam model"""

    def __init__(self, elastic_modulus, section_MOI, length, distributed_load, x_no_displ, x_no_rotation, x_free_end):

        self.ftype = 'float64'
        tf.keras.backend.set_floatx(self.ftype)
        
        # beam dimensions
        self.xmin = min(*x_no_displ, *x_no_rotation, *x_free_end)
        self.xmax = max(*x_no_displ, *x_no_rotation, *x_free_end)
        self.L = tf.constant(length, dtype=self.ftype)
        assert abs(self.L - (self.xmax - self.xmin)) < 1e-6
        # beam properties
        self.E = tf.constant(elastic_modulus, dtype=self.ftype)
        self.I = tf.constant(section_MOI, dtype=self.ftype)
        # beam loading
        self.q = tf.constant(distributed_load, dtype=self.ftype)
        # beam constraints
        self.x_fixed_w = tf.constant(x_no_displ, dtype=self.ftype)
        self.x_fixed_dwdx = tf.constant(x_no_rotation, dtype=self.ftype)
        self.x_free = tf.constant(x_free_end, dtype=self.ftype)

        # tensorflow model representation
        input_layer = keras.Input(shape=(1,))
        layer = SpatialPowers()(input_layer)
        output_layer = layers.Dense(1, activation='linear')(layer)
        self.tfmodel = keras.Model(inputs=input_layer, outputs=output_layer)

    @tf.function
    def calc_beam(self, x):
        with tf.GradientTape(persistent=True) as t:

            x = tf.transpose(tf.expand_dims(x, 0)) # treat each x value as a batch data point

            t.watch(x)

            # Change unit system to a different system that "happens" to have unit beam length.
            # Doing so vastly improves the training speed.
            L = self.L
            xn = x / L
            En = self.E * L * L
            In = self.I / (L * L * L * L)

            w = self.tfmodel(xn)
            dw_dx = t.gradient(w, xn)
            d2w_dx2 = t.gradient(dw_dx, xn)
            M = En * In * d2w_dx2
            V = t.gradient(M, xn)
            qm = t.gradient(V, xn)

        del t
        return (w, dw_dx, M, V, qm) 

    # @tf.function
    def eval_beam(self, x):

        # xtf = tf.constant(x, dtype=self.ftype) # TODO works without this line somehow, doesn't work with it somehow
        (w, _, M, V, _) = self.calc_beam(x)

        # change unit system back
        return (w*self.L, M*self.L, V)

    @tf.function
    def calc_loss(self, x_interior):
        with tf.GradientTape() as t:

            # evaluate the beam for all inputs at once
            x = tf.concat([self.x_fixed_w, self.x_fixed_dwdx, self.x_free, x_interior], axis=0)
            (w, dw_dx, M, V, qm) = self.calc_beam(x)

            # determine indices for each separated output, corresponding to each original input
            i0 = 0
            i1 = i0 + self.x_fixed_w.shape[0]
            i2 = i1 + self.x_fixed_dwdx.shape[0]
            i3 = i2 + self.x_free.shape[0]
            # get separated outputs
            w_fixed = w[i0:i1]
            dwdx_fixed = dw_dx[i1:i2]
            M_free = M[i2:i3]
            V_free = V[i2:i3]
            qm_diff_eq = qm[i3:]

            # compute losses
            qn = self.q * self.L # change unit system of q to consistenly compare the values
            loss_diff_eq = tf.reduce_sum(tf.square(qm_diff_eq - qn))
            loss_constraint = tf.reduce_sum(tf.square(w_fixed) + tf.square(dwdx_fixed))
            loss_free = tf.reduce_sum(tf.square(M_free) + tf.square(V_free))

            loss = loss_diff_eq + loss_constraint + loss_free

        grads = t.gradient(loss, self.tfmodel.trainable_variables)

        return (loss, grads, loss_diff_eq, loss_constraint, loss_free)

    def train(self):

        n_interior = 1 # number of interior points for computing the error in the beam differential equation q=EIw''''

        # get the initial gradients to apply, at some arbitrary initial x value
        loss, grads, _, _, _ = self.calc_loss([self.L/2])
        minloss = loss
        iter = 0
        rate = 1
        losstol = 1

        tic = time.perf_counter()

        while loss > 1e-4:
            optimizer = tf.keras.optimizers.Adam(learning_rate=rate, beta_1=0.9, beta_2=0.999)
            while loss > losstol:
                iter += 1
                optimizer.apply_gradients(zip(grads, self.tfmodel.trainable_variables))
                xrand = tf.random.uniform((n_interior,), minval=self.xmin, maxval=self.xmax, dtype=self.ftype)
                loss, grads, l1, l2, l3 = self.calc_loss(xrand)
                if loss > 100*minloss and optimizer.iterations > 100:   # If the loss starts _increasing_, we've overshot! Use a smaller learning rate.
                    break
                minloss = min(loss, minloss)
                #print("Iter: {}, Step: {}, Loss: {}, Internal: {}, Constraint: {}, Free: {}".format(iter, optimizer.iterations.numpy(), loss.numpy(), l1.numpy(), l2.numpy(), l3.numpy()))
                print("Iter: {}, Step: {}, Loss: {}".format(iter, optimizer.iterations.numpy(), loss.numpy()))
            rate /= 100
            if loss < losstol:  # We've reached a new low, so tighen the tolerance
                losstol /= 100

        toc = time.perf_counter()
        print('Time: {}'.format(toc-tic))
        print(self.tfmodel.trainable_variables[0].numpy())
