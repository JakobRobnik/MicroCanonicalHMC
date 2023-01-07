import numpy as np



class Quadratic():

    def __init__(self, d):

        self.d = d
        self.x0 = np.random.normal(d)


    def grad_f(self, x):
        return 0.5 * np.sum(np.square(x)), x



