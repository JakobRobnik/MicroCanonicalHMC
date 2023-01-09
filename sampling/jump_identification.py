import jax.numpy as jnp
import jax



def moments_to_score(x1a, x2a, Wa, x1b, x2b, Wb):
    """returns: log p(data | two regions) / p(data | one region) / effective number of data points"""
    var_a = x2a - jnp.square(x1a)
    var_b = x2b - jnp.square(x1b)
    x2 = (Wa * x2a + Wb * x2b) / (Wa + Wb)
    x1 = (Wa * x1a + Wb * x1b) / (Wa + Wb)
    var = x2 - jnp.square(x1)

    return jnp.log(var) - (Wa / (Wa + Wb)) * jnp.log(var_a) -  (Wb / (Wa + Wb)) * jnp.log(var_b)



def best_division(x, w, c, minn):

    x_sq = jnp.square(x)

    #initialize region A
    Wa = jnp.sum(w[:minn-1])
    x1a = jnp.sum(x[:minn-1] * w[:minn-1]) / Wa
    x2a = jnp.sum(x_sq[:minn-1] * w[:minn-1]) / Wa

    # initialize region B
    Wb = jnp.sum(w[minn+c-1:])
    x1b = jnp.sum(x[minn+c-1:] * w[minn+c-1:]) / Wb
    x2b = jnp.sum(x_sq[minn+c-1:] * w[minn+c-1:]) / Wb

    params0 = (x1a, x2a, Wa, x1b, x2b, Wb)

    def step(state, i):
        x1a, x2a, Wa, x1b, x2b, Wb = state

        #add i to region A
        x1a = (Wa * x1a + w[i] * x[i]) / (Wa + w[i])
        x2a = (Wa * x2a + w[i] * x_sq[i]) / (Wa + w[i])
        Wa = Wa + w[i]

        # remove i + c from region B
        x1b = (Wb * x1b - w[i+c] * x[i+c]) / (Wb - w[i+c])
        x2b = (Wb * x2b - w[i+c] * x_sq[i+c]) / (Wb - w[i+c])
        Wb = Wb - w[i]
        params = (x1a, x2a, Wa, x1b, x2b, Wb)

        return params, moments_to_score(*params)

    score = jax.lax.scan(step, params0, xs = jnp.arange(minn-1, len(x) - minn - c))[1] #do a scan over the time series

    index = jnp.argmax(score) + minn #the optimal choice

    x1a = jnp.average(x[:index], weights = w[:index])
    x1b = jnp.average(x[index+c:], weights=w[index+c:])

    X = x.at[jnp.concatenate((jnp.arange(index), jnp.arange(index+c, len(x))))].set(jnp.concatenate((x[:index] - x1a,  x[index+c:] - x1b)))
    W = w.at[jnp.arange(index, index+c)].set(-1.0)
    return X, W, score[index], index


def remove_jumps(x_given):
    """finds significant jumps in x and removes eliminates them"""

    c = 10             # transition region width which will be removed
    minn = 10          # minimum number of points in a region (such that the variance can be reliably estimated)
    max_iter = 10      # maximum number of jumps that will be removed
    min_score = jnp.log(2.0)  # minimum score for accepting the jump hypothesis ~= log(1 + (mu / 2 sigma)^2)


    x = jnp.copy(x_given)
    w = jnp.ones(len(x))

    to_do = [(0, len(x))]
    num_iter = 0

    while len(to_do) != 0 and num_iter < max_iter: #do the first remaining task on the to_do list untill there are none left

        imin, imax = to_do[0][0], to_do[0][1]
        X, W, score, index = best_division(x[imin:imax], w[imin:imax], c, minn)

        if score > min_score: # we accept the jump

            x = x.at[jnp.arange(imin, imax)].set(X)
            w = w.at[jnp.arange(imin, imax)].set(W)
            if index > 2 * minn + c + 1:
                to_do.append((imin, imin+index)) # add subintervals to the to_do list
            if imax - imin - index - c > 2 * minn + c + 1:
                to_do.append((imin + index + c, imax))

            num_iter += 1

        to_do.pop(0) #we accomplished the first task on the to_do list

    if num_iter != 0:
        print('Removed ' + str(num_iter) + ' jumps.')

    return x[w > -0.5]

