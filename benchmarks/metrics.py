
import sys

sys.path.append("./")
sys.path.append("../blackjax")

from blackjax.mcmc.adjusted_mclmc import rescale
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

import blackjax

from benchmarks.sampling_algorithms import (
    adjusted_hmc_no_tuning,
    adjusted_mclmc_tuning,
    calls_per_integrator_step,
    integrator_order,
    map_integrator_type_to_integrator,
    adjusted_mclmc,
    adjusted_mclmc_no_tuning,
    nuts,
    unadjusted_mclmc_no_tuning,
    target_acceptance_rate_of_order,
    unadjusted_mclmc,
    unadjusted_mclmc_tuning,
)

from benchmarks.sampling_algorithms import adjusted_mclmc_no_tuning, unadjusted_mclmc_no_tuning
import jax
import jax.numpy as jnp
import numpy as np
from blackjax.adaptation.adjusted_mclmc_adaptation import adjusted_mclmc_make_L_step_size_adaptation
from blackjax.adaptation.mclmc_adaptation import make_L_step_size_adaptation


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


def calculate_ess(err_t, grad_evals_per_step, num_tuning_steps, neff=100):

    grads_to_low, cutoff_reached = grads_to_low_error(
        err_t, grad_evals_per_step, 1.0 / neff
    )

    full_grads_to_low = grads_to_low 
    # + num_tuning_steps * grad_evals_per_step

    return (
        (neff / full_grads_to_low) * cutoff_reached,
        full_grads_to_low * (1 / cutoff_reached),
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


def grid_search_only_L(model, sampler, num_steps, num_chains, integrator_type, key, grid_size, opt='max', grid_iterations=2,):

    da_key, bench_key, init_pos_key, fast_tune_key = jax.random.split(key, 4)
    initial_position = model.sample_init(init_pos_key)

    if sampler=='adjusted_mclmc':

        integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

        L_proposal_factor = jnp.inf
        random_trajectory_length = True 
        if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps))
        else:
            integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps)

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=integrator,
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        sqrt_diag_cov=sqrt_diag_cov,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=model.logdensity_fn,
            L_proposal_factor=L_proposal_factor,
        )


        target_acc_rate = 0.9

        (
            blackjax_state_after_tuning,
            blackjax_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, fast_tune_key, model.logdensity_fn, False, target_acc_rate, kernel, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, params=None, max='avg', num_windows=2, tuning_factor=1.3)


    elif sampler=='mclmc':

        (blackjax_state_after_tuning, blackjax_sampler_params) = unadjusted_mclmc_tuning(
                    initial_position=initial_position,
                    num_steps=num_steps,
                    rng_key=fast_tune_key,
                    logdensity_fn=model.logdensity_fn,
                    integrator_type=integrator_type,
                    diagonal_preconditioning=False,
                    num_windows=2,
                )
    
    elif sampler=='adjusted_hmc':

        integrator = map_integrator_type_to_integrator["hmc"][integrator_type]

        random_trajectory_length = True 
        if random_trajectory_length:
            integration_steps_fn = lambda avg_num_integration_steps: lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)).astype(jnp.int32)
        else:
            integration_steps_fn = lambda avg_num_integration_steps: lambda _: jnp.ceil(avg_num_integration_steps).astype(jnp.int32)

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.dynamic_hmc.build_kernel(
        integrator=map_integrator_type_to_integrator["hmc"][integrator_type],
        integration_steps_fn=integration_steps_fn(avg_num_integration_steps),
        )(
            rng_key=rng_key,
            state=state,
            logdensity_fn=model.logdensity_fn,
            step_size=step_size,
            inverse_mass_matrix=jnp.diag(jnp.ones(model.ndims)),
        )

      

        (
            blackjax_state_after_tuning,
            blackjax_sampler_params) = adjusted_mclmc_tuning( initial_position, num_steps, rng_key=fast_tune_key, logdensity_fn=model.logdensity_fn,diagonal_preconditioning=False, target_acc_rate=0.9, kernel=kernel, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.1,  params=None, max='avg', num_windows=2,tuning_factor=1.3)
        
    else:
        raise Exception("sampler not recognized")
        

    z=blackjax_sampler_params.L
    state=blackjax_state_after_tuning
    jax.debug.print("initial L {x}", x=(z))

    # Lgrid = np.array([z])
    ESS = jnp.zeros(grid_size)
    ESS_AVG = jnp.zeros(grid_size)
    ESS_CORR_AVG = jnp.zeros(grid_size)
    ESS_CORR_MAX = jnp.zeros(grid_size)
    STEP_SIZE = jnp.zeros(grid_size)
    RATE = jnp.zeros(grid_size)
    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    for grid_iteration in range(grid_iterations):

        da_key_per_iter = jax.random.fold_in(da_key, grid_iteration)
        bench_key_per_iter = jax.random.fold_in(bench_key, grid_iteration)

        if grid_iteration==0:
            Lgrid = jnp.linspace(z/3, z * 3, grid_size)
        else:
            
            Lgrid = jnp.linspace(Lgrid[iopt-1], Lgrid[iopt+1], grid_size)
        jax.debug.print("Lgrid {x}", x=(Lgrid))

        for i in range(len(Lgrid)):
            da_key_per_iter = jax.random.fold_in(da_key_per_iter, i)
            bench_key_per_iter = jax.random.fold_in(bench_key_per_iter, i)
            jax.debug.print("L {x}", x=(Lgrid[i]))
            print("i", i)

            params = MCLMCAdaptationState(
                L=Lgrid[i],
                step_size=Lgrid[i]/5,
                sqrt_diag_cov=1.0,
            )

            if sampler in ['adjusted_mclmc', 'adjusted_hmc']:

                (
                    blackjax_state_after_tuning,
                    params,
                    _
                ) = adjusted_mclmc_make_L_step_size_adaptation(
                    kernel=kernel,
                    dim=model.ndims,
                    frac_tune1=0.1,
                    frac_tune2=0.0,
                    target=0.9,
                    diagonal_preconditioning=False,
                    fix_L_first_da=True,
                )(
                    state, params, num_steps, da_key_per_iter
                )

            elif sampler=='mclmc':
                    
                kernel = lambda sqrt_diag_cov: blackjax.mcmc.mclmc.build_kernel(
                    logdensity_fn=model.logdensity_fn,
                    integrator=integrator,
                    sqrt_diag_cov=sqrt_diag_cov,
                )
    

                    
                (
                    blackjax_state_after_tuning,
                    params,
                ) = make_L_step_size_adaptation(
                    kernel=kernel,
                    dim=model.ndims,
                    frac_tune1=0.1,
                    frac_tune2=0.0,
                    diagonal_preconditioning=False,
                )(
                    state, params, num_steps, da_key_per_iter
                )

            # elif sampler=='adjusted_hmc':


            # jax.debug.print("DA {x}", x=(final_da))
            jax.debug.print("benchmarking with L and step size {x}", x=(Lgrid[i], params.step_size))

            if sampler=='adjusted_mclmc':

                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
                    model,
                    adjusted_mclmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=blackjax_state_after_tuning,
                        sqrt_diag_cov=1.0,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        L_proposal_factor=jnp.inf,
                        return_ess_corr=False,
                        num_tuning_steps=0,
                    ),
                    bench_key_per_iter,
                    n=num_steps,
                    batch=num_chains,
                )
            
            elif sampler=='adjusted_hmc':

                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
                    model,
                    adjusted_hmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=blackjax_state_after_tuning,
                        sqrt_diag_cov=1.0,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        return_ess_corr=False,
                        num_tuning_steps=0,
                    ),
                    bench_key_per_iter,
                    n=num_steps,
                    batch=num_chains,
                )

            elif sampler=='mclmc':

                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark(
                    model,
                    unadjusted_mclmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=blackjax_state_after_tuning,
                        sqrt_diag_cov=1.0,
                        L=Lgrid[i],
                        step_size=params.step_size,
                        return_ess_corr=False,
                        num_tuning_steps=0,
                    ),
                    bench_key_per_iter,
                    n=num_steps,
                    batch=num_chains,
                )

            else:
                raise Exception("sampler not recognized")
                


            jax.debug.print("{x} ess", x=(ess))
            ESS = ESS.at[i].set(ess)
            ESS_AVG = ESS_AVG.at[i].set(ess_avg)
            # ESS_CORR_AVG[i] = ess_corr.mean().item()
            ESS_CORR_AVG = ESS_CORR_AVG.at[i].set(ess_corr.mean().item())
            # STEP_SIZE[i] = params.step_size.mean().item()
            STEP_SIZE = STEP_SIZE.at[i].set(params.step_size.mean().item())
            # RATE[i] = acceptance_rate.mean().item()
            RATE = RATE.at[i].set(acceptance_rate.mean().item())
        # iopt = np.argmax(ESS)
        if opt=='max':
            iopt = np.argmax(ESS)
        elif opt=='avg':
            iopt = np.argmax(ESS_AVG)
        else:
            raise Exception("opt not recognized")
        edge = grid_iteration==0 and (iopt == 0 or iopt == (len(Lgrid) - 1))

        print("iopt", iopt)
        jax.debug.print("optimal ess {x}", x=(ESS[iopt], ESS_AVG[iopt]))

    return Lgrid[iopt], STEP_SIZE[iopt], ESS[iopt], ESS_AVG[iopt], ESS_CORR_AVG[iopt], RATE[iopt], edge



def grid_search_only_stepsize(L, model, num_steps, num_chains, integrator_type, key, grid_size, grid_boundaries, state, opt='max'):


    step_size_grid = np.linspace(grid_boundaries[0], grid_boundaries[1], grid_size)
    # Lgrid = np.array([z])
    ESS = np.zeros_like(step_size_grid)
    ESS_AVG = np.zeros_like(step_size_grid)
    ESS_CORR_AVG = np.zeros_like(step_size_grid)
    ESS_CORR_MAX = np.zeros_like(step_size_grid)
    RATE = np.zeros_like(step_size_grid)
    # integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    da_key, bench_key = jax.random.split(key, 2)

    for i in range(len(step_size_grid)):

        

        # jax.debug.print("L {x}", x=params.L)
        # raise Exception

        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark(
            model,
            adjusted_mclmc_no_tuning(
                integrator_type=integrator_type,
                initial_state=state,
                sqrt_diag_cov=1.0,
                L=L,
                step_size=step_size_grid[i],
                L_proposal_factor=jnp.inf,
                return_ess_corr=True,
            ),
            bench_key,
            n=num_steps,
            batch=num_chains,
        )
        jax.debug.print("ess_avg {x}", x=(ess_avg))
        jax.debug.print("L {x}", x=(L))
        # jax.debug.print("{x} comparison of acceptance", x=(acceptance_rate, target_acc_rate))
        ESS[i] = ess
        ESS_AVG[i] = ess_avg
        ESS_CORR_AVG[i] = ess_corr.mean().item()
        ESS_CORR_MAX[i] = ess_corr.max().item()
        RATE[i] = acceptance_rate.mean().item()
    # iopt = np.argmax(ESS)
    if opt=='max':
        iopt = np.argmax(ESS)
    else:
        iopt = np.argmax(ESS_AVG)

    return step_size_grid[iopt], ESS[iopt], ESS_AVG[iopt], ESS_CORR_MAX[iopt], ESS_CORR_AVG[iopt], RATE[iopt]


def benchmark(model, sampler, key, n=10000, batch=None, pvmap=jax.pmap):


    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = pvmap(model.sample_init)(init_keys)  # [batch_size, dim_model]

    params, grad_calls_per_traj, acceptance_rate, expectation, ess_corr, num_tuning_steps = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=n, initial_position=pos, key=key
        )
    )(init_pos, keys)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)

    num_tuning_steps = jnp.mean(num_tuning_steps, axis=0)
    # jax.debug.print("{x} num tuning steps", x=num_tuning_steps)

    err_t_mean_avg = jnp.median(expectation[:, :, 0], axis=0)
    esses_avg, grads_to_low_avg, _ = calculate_ess(
        err_t_mean_avg, 
        grad_evals_per_step=avg_grad_calls_per_traj,
        num_tuning_steps=num_tuning_steps
    )

    err_t_mean_max = jnp.median(expectation[:, :, 1], axis=0)
    esses_max, _, _ = calculate_ess(
        err_t_mean_max, 
        grad_evals_per_step=avg_grad_calls_per_traj,
        num_tuning_steps=num_tuning_steps
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
