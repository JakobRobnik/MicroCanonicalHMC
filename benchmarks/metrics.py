
import jax
import jax.numpy as jnp
import numpy as np

def get_num_latents(target):
    return target.ndims


def err(f_true, var_f, contract):
    """Computes the error b^2 = (f - f_true)^2 / var_f
    Args:
        f: E_sampler[f(x)], can be a vector
        f_true: E_true[f(x)]
        var_f: Var_true[f(x)]
        contract: how to combine a vector f in a single number, can be for example jnp.average or jnp.max

    Returns:
        contract(b^2)
    """

    return jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))


def grads_to_low_error(err_t, grad_evals_per_step=1, low_error=0.01):
    """Uses the error of the expectation values to compute the effective sample size neff
    b^2 = 1/neff"""

    cutoff_reached = err_t[-1] < low_error
    return find_crossing(err_t, low_error) * grad_evals_per_step, cutoff_reached


def calculate_ess(err_t, grad_evals_per_step, neff=100):

    grads_to_low, cutoff_reached = grads_to_low_error(
        err_t, grad_evals_per_step, 1.0 / neff
    )

    return (
        (neff / grads_to_low) * cutoff_reached,
        grads_to_low * (1 / cutoff_reached),
        cutoff_reached,
    )


def find_crossing(array, cutoff):
    """the smallest M such that array[m] < cutoff for all m > M"""

    b = array > cutoff
    indices = jnp.argwhere(b)
    if indices.shape[0] == 0:
        print("\n\n\nNO CROSSING FOUND!!!\n\n\n", array, cutoff)
        return 1

    return jnp.max(indices) + 1


def cumulative_avg(samples):
    return jnp.cumsum(samples, axis=0) / jnp.arange(1, samples.shape[0] + 1)[:, None]

def grid_search(func, x, y, delta_x, delta_y, key, grid_size=5, num_iter=3,):
    """Args:
      func(x, y) = (score, extra_results),
      where score is the scalar that we would like to maximize (e.g. ESS averaged over the chains)
      and extra_results are some additional info that we would like to store, e.g. acceptance rate

      The initial grid will be set on the region [x - delta_x, x + delta x] \times [y - delta_y, y + delta y].
      In each iteration the grid will be shifted to the best location found by the previous grid and it will be shrinked,
      such that the nearest neigbours of the previous grid become the edges of the new grid.

      If at any point, the best point is found at the edge of the grid a warning is printed.

    Returns:
      (x, y, score, extra results) at the best parameters
    """

    def kernel(state, key):
        z, delta_z = state

        keys = jax.random.split(key, (grid_size, grid_size))

        # compute the func on the grid
        Z = np.linspace(z - delta_z, z + delta_z, grid_size)
        jax.debug.print("grid {x}", x=Z)
        Results = [[func(xx, yy, keys[i,j]) for (i, yy) in enumerate(Z[:, 1])] for (j,xx) in enumerate(Z[:, 0])]
        Scores = [
            [Results[i][j][0] for j in range(grid_size)] for i in range(grid_size)
        ]
        # grid = [[(xx, yy) for yy in Z[:, 1]] for xx in Z[:, 0]]
        # jax.lax.fori_loop(0, len(Results), lambda)
        # jax.debug.print("{x}",x="Outcomes from grid")
        # for i,f in enumerate(Scores):
        #     for j, g in enumerate(f):

                # jax.debug.print("{x}", x=(Scores[i][j].item(), grid[i][j][0].item(), grid[i][j][1].item()))

        # find the best point on the grid
        ind = np.unravel_index(np.argmax(Scores, axis=None), (grid_size, grid_size))

        if np.any(np.isin(np.array(ind), [0, grid_size - 1])):
            print("Best parameters found at the edge of the grid.")

        # new grid
        state = (
            np.array([Z[ind[i], i] for i in range(2)]),
            2 * delta_z / (grid_size - 1),
        )
    
        # sns.heatmap(np.array(Scores).T, annot=True, xticklabels=Z[:, 0], yticklabels=Z[:, 1])
        # plt.savefig(f"grid_search{iteration}.png")

        return (
            state,
            Results[ind[0]][ind[1]],
            np.any(np.isin(np.array(ind), [0, grid_size - 1])),
        )
    

    state = (np.array([x, y]), np.array([delta_x, delta_y]))

    initial_edge = False
    for iteration in range(num_iter):  # iteratively shrink and shift the grid
        key = jax.random.fold_in(key, iteration)
        state, results, edge = kernel(state, key)
        jax.debug.print("optimal result on iteration {x}", x=(iteration, results[0]))
        # jax.debug.print("Optimal params on iteration: {x}", x=(results[1]))
        # jax.debug.print("Optimal score on iteration: {x}", x=(results[0]))
        if edge and iteration == 0:
            initial_edge = True

    return [state[0][0], state[0][1], *results], initial_edge


def benchmark(model, sampler, key, n=10000, batch=None, pvmap=jax.pmap):


    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = pvmap(model.sample_init)(init_keys)  # [batch_size, dim_model]

    params, grad_calls_per_traj, acceptance_rate, expectation, ess_corr = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=n, initial_position=pos, key=key
        )
    )(init_pos, keys)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)

    err_t_mean_avg = jnp.median(expectation[:, :, 0], axis=0)
    esses_avg, grads_to_low_avg, _ = calculate_ess(
        err_t_mean_avg, grad_evals_per_step=avg_grad_calls_per_traj
    )

    err_t_mean_max = jnp.median(expectation[:, :, 1], axis=0)
    esses_max, _, _ = calculate_ess(
        err_t_mean_max, grad_evals_per_step=avg_grad_calls_per_traj
    )

    # if not jnp.isinf(jnp.mean(ess_corr)):

    #     jax.debug.print("{x} ESS CORR", x=ess_corr.mean())
        

    # ess_corr = jax.pmap(lambda x: effective_sample_size((jax.vmap(lambda x: ravel_pytree(x)[0])(x))[None, ...]))(samples)

    # print(ess_corr.shape,"shape")
    # ess_corr = jnp.mean(ess_corr, axis=0)

    # ess_corr = effective_sample_size(samples)
    # print("ess/n\n\n\n\n\n")
    # print(jnp.mean(ess_corr)/n)
    # print("ess/n\n\n\n\n\n")

    # flat_samples = jax.vmap(lambda x: ravel_pytree(x)[0])(samples)
    #     ess = effective_sample_size(flat_samples[None, ...])
    #     jax.debug.print("{x} INV ESS CORR",x=jnp.mean(1/ess))
    # jax.debug.print("{x}",x=jnp.mean(1/ess_corr))

    # return esses_max, esses_avg.item(), jnp.mean(1/ess_corr).item(), params, jnp.mean(acceptance_rate, axis=0), step_size_over_da
    return esses_max.item(), esses_avg.item(), ess_corr, params, jnp.mean(acceptance_rate, axis=0), grads_to_low_avg, err_t_mean_avg, err_t_mean_max
