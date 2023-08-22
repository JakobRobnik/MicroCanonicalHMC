import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize= (12, 8))

bshort = np.array([np.load('data/cauchy/nuts/' + str(i) + '_short.npy') for i in range(12)])

ff = 26
# By = np.sort(bshort, axis = 0)
# t = np.arange(1, 10**8, 10**5)
# plt.fill_between(t, By[2], By[7], color = 'tab:orange', alpha = 0.3)
# plt.plot(t, np.average([By[4], By[5]], axis = 0), color = 'tab:orange', lw = 4)


By = np.sort(bshort, axis = 0)
t = np.arange(1, 10001) * 1023
plt.fill_between(t, By[2], By[7], color = 'tab:red', alpha = 0.3)
plt.plot(t, np.average([By[4], By[5]], axis = 0), color = 'tab:red', lw = 4)


# b200 = VarH / 200
# plt.plot([0, 1e8], np.ones(2) * b200, '--', lw = 2, color = 'black')

plt.legend()
plt.yscale('log')
plt.xscale('log')
plt.ylabel(r'$b_{\mathcal{L}}^2$', fontsize = ff)
plt.xlabel(r'# gradient calls / $10^5$', fontsize = ff)
plt.legend(loc = 1, fontsize = ff)
plt.show()