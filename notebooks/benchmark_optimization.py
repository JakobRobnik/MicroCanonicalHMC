import matplotlib.pyplot as plt


from optimization import nesterov
from optimization.myProblems import *



optimizer = nesterov.Nesterov(Quadratic(100), 0.5)

loss = [optimizer.step() for i in range(30)]
plt.plot(loss, '.')

plt.xlabel('# gradient evaluations')
plt.ylabel('loss function')
plt.yscale('log')
plt.show()
