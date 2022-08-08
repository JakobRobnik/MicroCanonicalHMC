import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit  #I use numba, which makes the python evaluations significantly faster (as if it was a c code). If you do not want to download numba comment out the @njit decorators.
import sampler
import ESH
import torch as t

factorial = lambda n: 1 if n == 0 else n * factorial(n-1)


def compute_accuracy(step_names, hamiltonian):
    """"returns steps and diff
        diff[i, j, 0] = trajectory error for i-th integrator with steps[j] number of steps
        diff[i, j, 1] = relative energy error for i-th integrator with steps[j] number of steps
    """

    #properties of the trajectory
    total_time = 0.5 #total time of the trajectory
    x0, p0 = np.array([0.1, 0.3, 0.1]), np.array([0.2, 0.1, -0.15]) #initial condition


    #samplers
    if hamiltonian == 'Seperable':
        Samplers = [sampler.Seperable(nlogp_double_well, grad_nlogp_double_well, d, step=name) for name in step_names]

    else:
        Samplers = [sampler.Ruthless(nlogp_double_well, grad_nlogp_double_well, d, step=name) for name in step_names]


    #base solution: very small stepsize with the best method (the last one in the Samplers array)
    num_points, base = 7, 4
    Samplers[-1].dt = total_time/base**(num_points-1)
    E0 = Samplers[-1].hamiltonian(x0, p0)
    Samplers[-1].E = E0
    x_base, p_base = Samplers[-1].trajectory(x0, p0, base**(num_points-1))


    #main computation
    diff = np.empty((len(step_names), num_points, 2))
    for num_sampler in range(len(step_names)):
        #print('sampler ' + str(num_sampler))
        for n in range(num_points):
            Sampler =Samplers[num_sampler]
            dt = total_time / base ** n
            Sampler.dt = dt
            Sampler.E = E0

            #get the trajectory
            x, p = Sampler.trajectory(x0, Sampler.transform_initial_condition(x0, p0), base ** n)

            #results
            E_diff = np.abs(Sampler.hamiltonian(x[-1], Sampler.transform_to_synchronized(x[-1], p[-1])) - E0) / E0
            max_diff = np.max(np.sqrt(np.sum(np.square(x_base[::base**(num_points -1 - n), :] - x), axis = 1)))
            diff[num_sampler, n, 0] = max_diff
            diff[num_sampler, n, 1] = E_diff

    #stepsize = [total_time / base**n for n in range(1, nmax+1)]
    steps = np.power(base, np.arange(num_points))

    return steps, diff


def longterm_energy():

    # properties of the trajectory
    total_time = 500  # total time of the trajectory
    dt = 0.01
    total_steps = int(total_time / dt)
    time = np.arange(total_steps + 1) * dt
    x0, p0 = np.array([0.1, 0.3, -0.05]), np.array([0.214, 0.108, -0.12]) # initial condition

    Sampler1 = sampler.Seperable(nlogp_double_well, grad_nlogp_double_well, d, step='Yoshida')
    Sampler1.dt = dt
    Sampler2 = sampler.Seperable(nlogp_double_well, grad_nlogp_double_well, d, step='Leapfrog')
    Sampler2.dt = dt
    # Sampler1 = sampler.Ruthless(nlogp_double_well, grad_nlogp_double_well, d, step='RK4', dt=dt)
    # Sampler2 = sampler.Ruthless(nlogp_double_well, grad_nlogp_double_well, d, step='Leapfrog', dt=dt)

    E0 = Sampler1.hamiltonian(x0, p0)
    Sampler1.E, Sampler2.E = E0, E0

    x1, p1, = Sampler1.trajectory(x0, p0, total_steps)
    E1 = [np.abs(Sampler1.hamiltonian(x1[i], p1[i]) - E0)/np.abs(E0) for i in range(len(x1))]

    x2, p2, = Sampler2.trajectory(x0, Sampler2.transform_initial_condition(x0, p0), total_steps)
    E2 = [np.abs(Sampler2.hamiltonian(x2[i], Sampler2.transform_to_synchronized(x2[i], p2[i])) - E0)/np.abs(E0) for i in range(len(x2))]

    plt.plot(x1[:, 0], x1[:, 1])
    plt.show()
    plt.plot(time, x1[:, 0])
    plt.show()

    plt.plot(time, E1, '.', label = 'RK4')
    plt.plot(time, E2, '.', label='Leapfrog')
    plt.legend()
    plt.ylabel('(E(t) - E(0)) / E(0)')
    plt.xlabel('t')
    plt.yscale('log')
    plt.show()


def accuracy_plot():

    hamiltonian = (['Seperable', 'Ruthless'])[1]

    if hamiltonian == 'Seperable':
        labels = ['Symplectic Euler', 'Leapfrog', 'Yoshida']
        step_names = ['Euler', 'Leapfrog', 'Yoshida']
        order = [1.0, 2.0, 4.0]
        adjust = [1, 1, 3]

    if hamiltonian == 'Ruthless':
        labels = ['Symplectic Euler', 'EPSSP', 'Iterative Leapfrog', 'RK4']
        step_names = ['Euler', 'EPSSP', 'Leapfrog', 'RK4']
        order = [1.0, 2.0, 2.0, 4.0]
        adjust = [1, 6, 1, 4]

    steps, diff = compute_accuracy(step_names, hamiltonian)
    base, num_points = steps[1], len(steps)
    plt.figure(figsize=(15, 10))
    plt.suptitle('Integrator comparisson ('+hamiltonian+' Hamiltonian)')

    plt.subplot(1, 2, 1)
    for num_sampler in range(len(step_names)):
        plt.plot(adjust[num_sampler]*steps, diff[num_sampler, :, 0], '.:', label = labels[num_sampler])
        plt.plot(adjust[num_sampler]*steps, diff[num_sampler, num_points//2, 0]*np.power(steps/steps[num_points//2], -order[num_sampler]), color ='black', alpha = 0.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('# target distribution calls')
    plt.ylabel('trajectory deviation')
    plt.legend(loc = 3)

    plt.xlim(1, base**num_points)
    #plt.ylim(1e-16, 1)

    plt.subplot(1, 2, 2)
    for num_sampler in range(len(step_names)):
        plt.plot(adjust[num_sampler]*steps, diff[num_sampler, :, 1], '.:', label=labels[num_sampler])
        plt.plot(adjust[num_sampler]*steps, diff[num_sampler, num_points//2, 1] * np.power(steps / steps[num_points//2], -order[num_sampler]), color='black', alpha=0.5)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('# target distribution calls')
    plt.ylabel('energy deviation')
    plt.legend(loc = 3)

    plt.xlim(1, base**num_points)
    plt.ylim(1e-16, 1)

    plt.savefig('integrator_tst_'+hamiltonian+'.png')
    plt.show()



#@njit
def nlogp_gauss(x):
    """- log p of the target distribution"""
    return 0.5 * np.sum(np.square(x))
#@njit
def grad_nlogp_gauss(x):
    return x


#@njit
def nlogp_double_well(x):
    a, b, c = params[0], params[1], params[2]
    return b * x[0]**4 - x[0]**2 - a*x[0]**3 - c + x[1]**2 + 0.5*x[2]**2

#@njit
def grad_nlogp_double_well(x):
    a, b, c = params[0], params[1], params[2]
    return np.array([4*b*x[0]**3 - 2*x[0] - 3*a*x[0]**2, 2*x[1], x[2]])

def orbits():
    d = 2
    # t.manual_seed(1)
    np.random.seed(1)
    x0 = np.random.normal(size = d)#.tolist()
    #esh
    # x1, p0 = esh.trajectory(x0, 0.5, 41)
    # print('ESH steps: ' + str(len(x1)))
    samp = ESH.esh(nlogp_gauss, grad_nlogp_gauss, d, 1.5)
    t, x1 = samp.sample(x0, 0.8*10, 1)

    np.random.seed(0)
    p0 = np.random.normal(size = d)
    p0 /= np.sqrt(np.sum(np.square(p0)))

    #"ground truth"
    ruthless = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step='RK4')
    ruthless.E = ruthless.hamiltonian(x0, p0)
    ruthless.dt = 0.0001
    x00, p00 = ruthless.trajectory_fixed_time(x0, p0, 26300 * ruthless.dt)
    #v_initial = np.sqrt(np.sum(np.square(x00[1] - x00[0]))) / ruthless.dt
    #total_time = 2.6


    # ruthless = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step='RK4')
    # ruthless.E = ruthless.hamiltonian(x0, p0)
    # ruthless.dt = 0.06
    # x2, p2 = ruthless.trajectory_fixed_time(x0, ruthless.transform_initial_condition(x0, p0), total_time)
    # print('RK4 steps: ' + str(len(x2)))
    #
    # ruthless = sampler.Ruthless(nlogp_gauss, grad_nlogp_gauss, d, step='Adaptive Leapfrog')
    # ruthless.E = ruthless.hamiltonian(x0, p0)
    # dt_initial = 0.011
    # ruthless.dx = v_initial * dt_initial  # initial momentum is normalized to 1
    # ruthless.dt = dt_initial
    # x3, p3 = ruthless.trajectory_fixed_time(x0, ruthless.transform_initial_condition(x0, p0), total_time)
    # print('AIL steps: ' + str(len(x3)))

    plt.plot(x1[:, 0], x1[:, 1], '.', color = 'tab:blue', label = r'ESP ($\epsilon = 0.5$)')
    #plt.plot(x2[:, 0], x2[:, 1], '.', color = 'tab:red', label = 'myESP')
    # plt.plot(x3[:, 0], x3[:, 1], '.', color = 'tab:purple', label = 'AIL')

    plt.plot(x00[:, 0], x00[:, 1], ':', color = 'black', label = 'ground truth')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    #plt.savefig('trajectory integrators.png')
    plt.show()


d = 3
b = 0.1
a = 0.03
xmin = (3 * a + np.sqrt(9 * a * a + 32 * b)) / (8 * b)
c = xmin ** 2 * (b * xmin ** 2 - a * xmin - 1)
params = [a, b, c]

orbits()
#longterm_energy()

