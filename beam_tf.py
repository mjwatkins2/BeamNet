from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

from model import BeamModel

# disable GPU because this is too small to benefit
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

if __name__ == '__main__':

    E = 10.0e3  # 10,000 ksi
    I = 50      # 50 in^4
    L = 60      # 60 in 
    q = -0.01   # 0.010 kip/in

    xLeft = [0]
    xRight = [L]

    model = BeamModel(E, I, L, q, xLeft, xLeft, xRight)
    model.train()

    xplot = np.linspace(0, L, 100).T
    (w, M, V) = model.eval_beam(xplot)
    fig, axs = plt.subplots(3, 1)
    (ax1, ax2, ax3) = axs
    ax1.plot(xplot, w.numpy())
    ax1.set(ylabel='Displacement (w)')
    ax2.plot(xplot, V.numpy())
    ax2.set(ylabel='Shear (V)')
    ax3.plot(xplot, M.numpy())
    ax3.set(ylabel='Bending Moment (M)')
    for ax in axs:
        ax.set(xlabel='x')
    plt.show()