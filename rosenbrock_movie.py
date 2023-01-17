import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import pandas as pd
import mchmc
import myHMC
from benchmark_targets import *

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage



### sampling ###
target = Rosenbrock(d = 100, Q= 0.1)
target.prior_draw = lambda key: jax.random.normal(key, shape = (target.d, ), dtype = 'float64') * 4 +2

eps = 0.2
sampler = mchmc.Sampler(target, 70 * eps, eps, 'LF', False)

steps = 500
particles = 500
X, W = sampler.parallel_sample(particles, steps)

#shape(X) = (#particles, #steps, #dimensions)



time = np.arange(steps)
x_samples = X[:, :, 0]
y_samples = X[:, :, target.d//2]

#L = [[target.nlogp(X[i, j, :]) for j in range(len(X[1]))] for i in range(len(X))]
# for i in range(100):
#     plt.plot(L[i])
#
# plt.plot(L[97], color = 'red', lw = 5)
# plt.show()
#
#
# exit()

# mask = (x_samples[:, -1] > 2) & (y_samples[:, -1] > 10)
# for i in range(100):
#     if mask[i]:
#         print(i)
# exit()

xmin, xmax = -3, 5#np.min(x_samples) * 1.1, np.max(x_samples) * 1.1
ymin, ymax = -5, 13#np.min(y_samples) * 1.1, np.max(y_samples) * 1.1

fig = plt.figure(figsize=((xmax-xmin)*0.7, (ymax-ymin)*0.7))
ax = plt.subplot()


### background ###
num = 100
x = np.linspace(xmin, xmax, num)
y = np.linspace(ymin, ymax, num)
xmesh, ymesh = np.meshgrid(x, y)
Q = target.Q
zmesh = np.exp(-0.5 * (jnp.square(xmesh - 1.0) + jnp.square(jnp.square(xmesh) - ymesh) / Q))



### animation ###

duration = 10 #s

def make_frame(time_in_movie):
    index = np.argmin(np.abs(time/steps - time_in_movie / duration))
    ax.cla()
    ax.contourf(xmesh, ymesh, zmesh, cmap='cividis')
    ax.plot(x_samples[:, index], y_samples[:, index], '.', color='red')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    # returning numpy image
    return mplfig_to_npimage(fig)


animation = VideoClip(make_frame, duration=duration)

# displaying animation with auto play and looping
# animation.ipython_display(fps=30, loop=True, autoplay=True)
animation.write_gif('rosenbrock_Q=0.1_bounces70.gif', fps=15)





