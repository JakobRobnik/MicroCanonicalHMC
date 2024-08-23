import sys

sys.path.append("./")

from blackjax.diagnostics import effective_sample_size
from collections import defaultdict
from functools import partial
import math
import operator
import os
import pprint
from statistics import mean, median
import jax
import jax.numpy as jnp
import pandas as pd
import scipy
from jax.flatten_util import ravel_pytree


from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=" + str(128)
num_cores = jax.local_device_count()
# print(num_cores, jax.lib.xla_bridge.get_backend().platform)

import itertools

import numpy as np
from blackjax.mcmc.adjusted_mclmc import rescale

import blackjax
from benchmarks.sampling_algorithms import (
    adjusted_mclmc_tuning,
    calls_per_integrator_step,
    integrator_order,
    map_integrator_type_to_integrator,
    run_adjusted_mclmc,
    run_adjusted_mclmc_no_tuning,
    run_nuts,
    run_unadjusted_mclmc_no_tuning,
    target_acceptance_rate_of_order,
    run_unadjusted_mclmc,
    unadjusted_mclmc_tuning,
)

# run_adjusted_mclmc, run_nuts, samplers
from benchmarks.inference_models import (
    Banana,
    Brownian,
    Funnel,
    GermanCredit,
    IllConditionedGaussian,
    ItemResponseTheory,
    MixedLogit,
    StandardNormal,
    StochasticVolatility,
)
from blackjax.mcmc.integrators import (
    generate_euclidean_integrator,
    generate_isokinetic_integrator,
    isokinetic_mclachlan,
    mclachlan_coefficients,
    omelyan_coefficients,
    velocity_verlet,
    velocity_verlet_coefficients,
    yoshida_coefficients,
)
from blackjax.util import run_inference_algorithm, store_only_expectation_values


models = {
    ## Rosenbrock(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000}, # no Ex2
    # Cauchy(100) : {'mclmc': 2000, 'adjusted_mclmc' : 2000, 'nuts': 2000},
    Brownian(): {"mclmc": 20000, "adjusted_mclmc": 20000, "nuts": 4000},
    # StandardNormal(2) : {'mclmc': 1000, 'adjusted_mclmc' : 1000, 'nuts': 1000},
    # StandardNormal(50) : {'mclmc': 800, 'adjusted_mclmc' : 800, 'nuts': 800},
    # StandardNormal(100) : {'mclmc': 800, 'adjusted_mclmc' : 800, 'nuts': 800},
    # StandardNormal(500) : {'mclmc': 800, 'adjusted_mclmc' : 800, 'nuts': 800},
    # StandardNormal(1000) : {'mclmc': 800, 'adjusted_mclmc' : 800, 'nuts': 800},
    # Banana() : {'mclmc': 10000, 'adjusted_mclmc' : 10000, 'nuts': 10000},
    # Funnel() : {'mclmc': 20000, 'adjusted_mclmc' : 80000, 'nuts': 40000},
    # Banana() : {'mclmc': 10000, 'adjusted_mclmc' : 10000, 'nuts': 10000},
    # IllConditionedGaussian(100, 100):   {'mclmc': 20000, 'adjusted_mclmc' : 20000, 'nuts': 20000},
    # GermanCredit(): {'mclmc': 80000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    # ItemResponseTheory(): {'mclmc': 20000, 'adjusted_mclmc' : 40000, 'nuts': 20000},
    # StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000}
}

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


# def gridsearch_tune(
#     key,
#     iterations,
#     grid_size,
#     model,
#     sampler,
#     batch,
#     num_steps,
#     center_L,
#     center_step_size,
#     contract,
# ):
#     results = defaultdict(float)
#     converged = False
#     keys = jax.random.split(key, iterations + 1)
#     for i in range(iterations):
#         print(f"EPOCH {i}")
#         width = 2
#         step_sizes = np.logspace(
#             np.log10(center_step_size / width),
#             np.log10(center_step_size * width),
#             grid_size,
#         )
#         Ls = np.logspace(np.log10(center_L / 2), np.log10(center_L * 2), grid_size)

#         grid_keys = jax.random.split(keys[i], grid_size ^ 2)
#         print(f"center step size {center_step_size}, center L {center_L}")
#         for j, (step_size, L) in enumerate(itertools.product(step_sizes, Ls)):
#             ess, grad_calls_until_convergence, _, _, _ = benchmark_chains(
#                 model,
#                 sampler(step_size=step_size, L=L),
#                 grid_keys[j],
#                 n=num_steps,
#                 batch=batch,
#                 contract=contract,
#             )
#             results[(step_size, L)] = (ess, grad_calls_until_convergence)

#         best_ess, best_grads, (step_size, L) = max(
#             [(results[r][0], results[r][1], r) for r in results],
#             key=operator.itemgetter(0),
#         )
#         # raise Exception
#         print(
#             f"best params on iteration {i} are stepsize {step_size} and L {L} with Grad Calls until Convergence {best_grads}"
#         )
#         if L == center_L and step_size == center_step_size:
#             print("converged")
#             converged = True
#             break
#         else:
#             center_L, center_step_size = L, step_size

#     pprint.pp(results)
#     # print(f"best params on iteration {i} are stepsize {step_size} and L {L} with Grad Calls until Convergence {best_grads}")
#     # print(f"L from ESS (0.4 * step_size/ESS): {0.4 * step_size/best_ess}")
#     return center_L, center_step_size, converged


def grid_search(func, x, y, delta_x, delta_y, size_grid=5, num_iter=3):
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

    def kernel(state):
        z, delta_z = state

        # compute the func on the grid
        Z = np.linspace(z - delta_z, z + delta_z, size_grid)
        # jax.debug.print("grid {x}", x=Z)
        Results = [[func(xx, yy) for yy in Z[:, 1]] for xx in Z[:, 0]]
        Scores = [
            [Results[i][j][0] for j in range(size_grid)] for i in range(size_grid)
        ]
        grid = [[(xx, yy) for yy in Z[:, 1]] for xx in Z[:, 0]]
        # jax.lax.fori_loop(0, len(Results), lambda)
        # jax.debug.print("{x}",x="Outcomes from grid")
        for i,f in enumerate(Scores):
            for j, g in enumerate(f):

                jax.debug.print("{x}", x=(Scores[i][j].item(), grid[i][j][0].item(), grid[i][j][1].item()))

        # find the best point on the grid
        ind = np.unravel_index(np.argmax(Scores, axis=None), (size_grid, size_grid))

        if np.any(np.isin(np.array(ind), [0, size_grid - 1])):
            print("Best parameters found at the edge of the grid.")

        # new grid
        state = (
            np.array([Z[ind[i], i] for i in range(2)]),
            2 * delta_z / (size_grid - 1),
        )

        return (
            state,
            Results[ind[0]][ind[1]],
            np.any(np.isin(np.array(ind), [0, size_grid - 1])),
        )

    state = (np.array([x, y]), np.array([delta_x, delta_y]))

    initial_edge = False
    for iteration in range(num_iter):  # iteratively shrink and shift the grid
        state, results, edge = kernel(state)
        jax.debug.print("Optimal params on iteration: {x}", x=(results[1]))
        jax.debug.print("Optimal score on iteration: {x}", x=(results[0]))
        if edge and iteration == 0:
            initial_edge = True

    return [state[0][0], state[0][1], *results], initial_edge


def benchmark_chains(model, sampler, key, n=10000, batch=None):

    pvmap = jax.pmap

    d = get_num_latents(model)
    if batch is None:
        batch = np.ceil(1000 / d).astype(int)
    key, init_key = jax.random.split(key, 2)
    keys = jax.random.split(key, batch)

    init_keys = jax.random.split(init_key, batch)
    init_pos = pvmap(model.sample_init)(init_keys)  # [batch_size, dim_model]

    params, grad_calls_per_traj, acceptance_rate, expectation = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=n, initial_position=pos, key=key
        )
    )(init_pos, keys)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)
    # try:
    #     print(jnp.nanmean(params.step_size, axis=0), jnp.nanmean(params.L, axis=0))
    # except:
    #     pass

    # jax.vmap(lambda f: contract(jnp.square(f - f_true) / var_f))
    # raise Exception

    # full_avg = lambda x : err(model.E_x2, model.Var_x2, jnp.average)(cumulative_avg(x**2))
    # full_max = lambda x : err(model.E_x2, model.Var_x2, jnp.max)(cumulative_avg(x**2))
    # err_t = pvmap(err(model.E_x2, model.Var_x2, contract))(samples)
    # err_t_avg = pvmap(full_avg)(samples)
    # err_t_max = pvmap(full_max)(samples)
    # jax.debug.print("expectation {x} ", x=expectation[0][:4])
    # jax.debug.print("full avg {x} ", x=err_t_avg[0][:4])
    # print(expectation.shape, "expectation shape")
    # print(err_t_avg.shape, "err_t_avg shape")
    # print(expectation.shape, "expectation shape")
    err_t_median_avg = jnp.median(expectation[:, :, 0], axis=0)
    esses_avg, _, _ = calculate_ess(
        err_t_median_avg, grad_evals_per_step=avg_grad_calls_per_traj
    )

    err_t_median_max = jnp.median(expectation[:, :, 1], axis=0)
    esses_max, _, _ = calculate_ess(
        err_t_median_max, grad_evals_per_step=avg_grad_calls_per_traj
    )

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
    return esses_max, esses_avg.item(), None, params, jnp.mean(acceptance_rate, axis=0)


def benchmark_no_chains(
    model,
    sampler,
    key,
    n=10000,
    contract=jnp.average,
):

    d = get_num_latents(model)
    key, init_key = jax.random.split(key, 2)
    init_pos = model.sample_init(init_key)  # [batch_size, dim_model]

    (
        ex2_empirical,
        params,
        grad_calls_per_traj,
        acceptance_rate,
        step_size_over_da,
        final_da,
    ) = sampler(
        logdensity_fn=model.logdensity_fn,
        num_steps=n,
        initial_position=init_pos,
        transform=model.transform,
        key=key,
    )
    avg_grad_calls_per_traj = grad_calls_per_traj
    try:
        print(jnp.nanmean(params.step_size, axis=0), jnp.nanmean(params.L, axis=0))
    except:
        pass

    err_t = (err(model.E_x2, model.Var_x2, contract))(ex2_empirical)

    # outs = [calculate_ess(b, grad_evals_per_step=avg_grad_calls_per_traj) for b in err_t]
    # # print(outs[:10])
    # esses = [i[0].item() for i in outs if not math.isnan(i[0].item())]
    # grad_calls = [i[1].item() for i in outs if not math.isnan(i[1].item())]
    # return(mean(esses), mean(grad_calls))
    # print(final_da.mean(), "final da")

    err_t_median = err_t  # jnp.median(err_t, axis=0)
    # import matplotlib.pyplot as plt
    # plt.plot(np.arange(1, 1+ len(err_t_median))* 2, err_t_median, color= 'teal', lw = 3)
    # plt.xlabel('gradient evaluations')
    # plt.ylabel('average second moment error')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.savefig('brownian.png')
    # plt.close()
    esses, grad_calls, _ = calculate_ess(
        err_t_median, grad_evals_per_step=avg_grad_calls_per_traj
    )
    return esses, grad_calls, params, acceptance_rate, step_size_over_da



def benchmark_mhmchmc(batch_size):

    key0, key1 = jax.random.split(jax.random.PRNGKey(5), 2)

    integrators = ["mclachlan"]
    for model in models:
        results = defaultdict(tuple)
        for integrator_type in integrators:
            num_chains = batch_size  # 1 + batch_size//model.ndims
            print(
                f"NUMBER OF CHAINS for {model.name} and adjusted_mclmc is {num_chains}"
            )
            num_steps = models[model]["adjusted_mclmc"]
            print(f"NUMBER OF STEPS for {model.name} and MHCMLMC is {num_steps}")

            grid = True
            if grid:
                ####### run adjusted_mclmc with standard tuning + grid search

                (
                    init_pos_key,
                    tune_key,
                    grid_key,
                    bench_key,
                    unadjusted_grid_key,
                    unadjusted_bench_key,
                ) = jax.random.split(key1, 6)




                initial_position = model.sample_init(init_pos_key)

                (
                    blackjax_state_after_tuning,
                    blackjax_adjusted_mclmc_sampler_params,
                ) = adjusted_mclmc_tuning(
                    initial_position=initial_position,
                    num_steps=num_steps,
                    rng_key=tune_key,
                    logdensity_fn=model.logdensity_fn,
                    integrator_type=integrator_type,
                    frac_tune3=0.0,
                    target_acc_rate=0.9,
                    diagonal_preconditioning=False,
                )



                do_grid_search_for_adjusted = True
                if do_grid_search_for_adjusted:
                    print(
                        f"target acceptance rate {target_acceptance_rate_of_order[integrator_order(integrator_type)]}"
                    )
                    print(
                        f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
                    )

                    L_proposal_factor = jnp.inf
                    def func(L, step_size):

                        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                            model=model,
                            sampler=run_adjusted_mclmc_no_tuning(
                                integrator_type=integrator_type,
                                initial_state=blackjax_state_after_tuning,
                                sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                                L=L,
                                step_size=step_size,
                                L_proposal_factor=L_proposal_factor,
                            ),
                            key=grid_key,
                            n=num_steps,
                            batch=batch_size,
                        )

                        return ess, (params.L.mean(), params.step_size.mean())

                    out, edge = grid_search(
                        func=func,
                        # x=blackjax_adjusted_mclmc_sampler_params.L,
                        # y=blackjax_adjusted_mclmc_sampler_params.step_size,
                        x=3.316967,
                        y=0.390205,
                        delta_x=blackjax_adjusted_mclmc_sampler_params.L - 0.2,
                        delta_y=blackjax_adjusted_mclmc_sampler_params.step_size - 0.2,
                        size_grid=6,
                        num_iter=4,
                    )


                    print("BENCHMARK after finding optimal params with grid \n\n\n")
                    ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                        model,
                        run_adjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_state_after_tuning,
                        ),
                        bench_key,
                        n=num_steps,
                        batch=num_chains,
                    )
                   

                    results[
                        (
                            model.name,
                            model.ndims,
                            f"mhmchmc:grid_edge{edge}",
                            jnp.nanmean(params.L).item(),
                            jnp.nanmean(params.step_size).item(),
                            integrator_type,
                            f"gridsearch",
                            acceptance_rate.mean().item(),
                            False,
                            1 / L_proposal_factor,
                            ess_avg,
                            ess_corr,
                            num_steps,
                        )
                    ] = ess.item()

                

            
            
            ####### run mclmc with standard tuning
            for preconditioning in [False]:

                unadjusted_with_tuning_key, adjusted_with_tuning_key, adjusted_with_tuning_key_stage3, nuts_key_with_tuning = jax.random.split(key0, 4)


                if True:
                    
                    ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                        model,
                        run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=False),
                        unadjusted_with_tuning_key,
                        n=num_steps,
                        batch=num_chains,
                    )

                    
                    results[
                        (
                            model.name, model.ndims, "mclmc", params.L.mean().item(), params.step_size.mean().item(), (integrator_type), "standard", 1.0, preconditioning, 0, ess_avg, ess_corr, num_steps,
                        )
                    ] = ess.item()
                    print(f"mclmc with tuning ESS {ess}")

                    ####### run adjusted_mclmc with standard tuning
                    for target_acc_rate, L_proposal_factor in itertools.product(
                        [0.9], [jnp.inf]
                    ):  # , 3., 1.25, 0.5] ):
                        # coeffs = mclachlan_coefficients
                        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                            model,
                            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.0, L_proposal_factor=L_proposal_factor,
                            target_acc_rate=target_acc_rate),
                            adjusted_with_tuning_key,
                            n=num_steps,
                            batch=num_chains,
                            
                        )
                        results[
                            (
                                model.name,
                                model.ndims,
                                "mhmclmc:" + str(target_acc_rate),
                                jnp.nanmean(params.L).item(),
                                jnp.nanmean(params.step_size).item(),
                                (integrator_type),
                                "standard",
                                acceptance_rate.mean().item(),
                                preconditioning,
                                1 / L_proposal_factor,
                                ess_avg,
                                ess_corr,
                                num_steps,
                            )
                        ] = ess.item()
                        print(f"adjusted_mclmc with tuning ESS {ess}")

                        # integrator_type = mclachlan_coefficients
                        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                            model,
                            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.1, L_proposal_factor=L_proposal_factor,
                            target_acc_rate=target_acc_rate,),
                            adjusted_with_tuning_key_stage3,
                            n=num_steps,
                            batch=num_chains,
                        )
                        results[
                            (
                                model.name,
                                model.ndims,
                                "mhmclmc:st3:" + str(target_acc_rate),
                                jnp.nanmean(params.L).item(),
                                jnp.nanmean(params.step_size).item(),
                                (integrator_type),
                                "standard",
                                acceptance_rate.mean().item(),
                                preconditioning,
                                1 / L_proposal_factor,
                                ess_avg,
                                ess_corr,
                                num_steps,
                            )
                        ] = ess.item()
                        print(f"adjusted_mclmc with tuning ESS {ess}")

                ####### run nuts

                
                ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
                    model,
                    run_nuts(integrator_type="velocity_verlet", preconditioning=False),
                    nuts_key_with_tuning,
                    n=num_steps,
                    batch=num_chains,
                )
                
                results[
                    (
                        model.name,
                        model.ndims,
                        "nuts",
                        0.0,
                        0.0,
                        (integrator_type),
                        "standard",
                        acceptance_rate.mean().item(),
                        preconditioning,
                        0,
                        ess_avg,
                        ess_corr,
                        num_steps,
                    )
                ] = ess.item()

        df = pd.Series(results).reset_index()
        df.columns = [
            "model", "dims", "sampler", "L", "step_size", "integrator", "tuning", "acc_rate", "preconditioning", "inv_L_prop", "ess_avg", "inv_ess_corr", "num_steps", "ESS"]
        # df.result = df.result.apply(lambda x: x[0].item())
        # df.model = df.model.apply(lambda x: x[1])
        df.to_csv(f"results{model.name}{model.ndims}.csv", index=False)

    return results


# TODO: not updated to new code yet!
def benchmark_omelyan(batch_size):

    key = jax.random.PRNGKey(2)
    results = defaultdict(tuple)
    for variables in itertools.product(
        # ["adjusted_mclmc", "nuts", "mclmc", ],
        [
            StandardNormal(d)
            for d in np.ceil(np.logspace(np.log10(1e1), np.log10(1e2), 2)).astype(int)
        ],
        # [StandardNormal(d) for d in np.ceil(np.logspace(np.log10(10), np.log10(10000), 5)).astype(int)],
        # models,
        # [velocity_verlet_coefficients, mclachlan_coefficients, yoshida_coefficients, omelyan_coefficients],
        ["mclachlan", "omelyan"],
    ):

        model, integrator_type = variables

        # num_chains = 1 + batch_size//model.ndims
        num_chains = batch_size

        current_key, key = jax.random.split(key)
        init_pos_key, init_key, tune_key, bench_key, grid_key = jax.random.split(
            current_key, 5
        )

        # num_steps = models[model][sampler]

        num_steps = 1000

        initial_position = model.sample_init(init_pos_key)

        initial_state = blackjax.mcmc.adjusted_mclmc.init(
            position=initial_position,
            logdensity_fn=model.logdensity_fn,
            random_generator_arg=init_key,
        )

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
            integrator=map_integrator_type_to_integrator["mclmc"][integrator_type],
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * rescale(avg_num_integration_steps)
            ),
            sqrt_diag_cov=sqrt_diag_cov,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=model.logdensity_fn,
        )

        (state, blackjax_adjusted_mclmc_sampler_params, _, _) = (
            blackjax.adjusted_mclmc_find_L_and_step_size(
                mclmc_kernel=kernel,
                num_steps=num_steps,
                state=initial_state,
                rng_key=tune_key,
                target=target_acceptance_rate_of_order[
                    integrator_order(integrator_type)
                ],
                # frac_tune1=0.1,
                # frac_tune2=0.1,
                # frac_tune3=0.1,
                diagonal_preconditioning=False,
            )
        )

        print(
            f"\nModel: {model.name,model.ndims}, Sampler: Adjusted MCLMC\n Integrator: {integrator_type}\nNumber of chains {num_chains}",
        )
        print(
            f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
        )

        # ess, grad_calls, _ , _ = benchmark_chains(model, run_adjusted_mclmc_no_tuning(integrator_type=coefficients, L=blackjax_mclmc_sampler_params.L, step_size=blackjax_mclmc_sampler_params.step_size, sqrt_diag_cov=1.),bench_key_pre_grid, n=num_steps, batch=num_chains, contract=jnp.average)

        # results[((model.name, model.ndims), sampler, (coefficients), "without grid search")] = (ess, grad_calls)

        # L, step_size, converged = gridsearch_tune(grid_key, iterations=10, contract=jnp.average, grid_size=5, model=model, sampler=partial(run_adjusted_mclmc_no_tuning, integrator_type=integrator_type, initial_state=state, sqrt_diag_cov=1.), batch=num_chains, num_steps=num_steps, center_L=blackjax_adjusted_mclmc_sampler_params.L, center_step_size=blackjax_adjusted_mclmc_sampler_params.step_size)
        # print(f"params after grid tuning are L={L}, step_size={step_size}")

        def func(L, step_size):

            r = benchmark_chains(
                model=model,
                sampler=run_adjusted_mclmc_no_tuning(
                    integrator_type=integrator_type,
                    initial_state=state,
                    sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                    L=L,
                    step_size=step_size,
                    L_proposal_factor=jnp.inf,
                ),
                key=grid_key,
                n=num_steps,
                batch=batch_size,
            )

            return r[0], r[4]

        out, edge = grid_search(
            func=func,
            x=blackjax_adjusted_mclmc_sampler_params.L,
            y=blackjax_adjusted_mclmc_sampler_params.step_size,
            delta_x=blackjax_adjusted_mclmc_sampler_params.L - 0.2,
            delta_y=blackjax_adjusted_mclmc_sampler_params.step_size - 0.2,
            size_grid=3,
            num_iter=1,
        )

        # results[(model.name, model.ndims, "mhmchmc:grid_new", out[0].item(), out[1].item(), integrator_type, f"gridsearch", out[3].item(), True, 1/L_proposal_factor, "n/a", "n/a", num_steps)] = out[2].item()

        print("BENCHMARK after finding optimal params with grid \n\n\n")
        ess, ess_avg, ess_corr, params, acceptance_rate, _ = benchmark_chains(
            model,
            run_adjusted_mclmc_no_tuning(
                integrator_type=integrator_type,
                initial_state=state,
                sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                L=out[0],
                step_size=out[1],
                L_proposal_factor=jnp.inf,
            ),
            bench_key,
            n=num_steps,
            batch=num_chains,
        )

        # ess, grad_calls, _ , _, _ = benchmark_chains(model, run_adjusted_mclmc_no_tuning(integrator_type=integrator_type, L=L, step_size=step_size, sqrt_diag_cov=1., initial_state=state),bench_key, n=num_steps, batch=num_chains, contract=jnp.average)

        # print(f"grads to low bias: {grad_calls}")

        results[
            (
                model.name,
                model.ndims,
                "mclmc",
                (integrator_type),
                edge,
                out[0].item(),
                out[1].item(),
            )
        ] = ess.item()

    save = True
    if save:

        df = pd.Series(results).reset_index()
        df.columns = [
            "model",
            "dims",
            "sampler",
            "integrator",
            "convergence",
            "L",
            "step_size",
            "ESS",
        ]
        # df.result = df.result.apply(lambda x: x[0].item())
        # df.model = df.model.apply(lambda x: x[1])
        df.to_csv("omelyan.csv", index=False)





def test_new_run_inference():

    init_key, state_key, run_key = jax.random.split(jax.random.PRNGKey(0), 3)
    model = StandardNormal(2)
    initial_position = model.sample_init(init_key)
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
    )
    integrator_type = "mclachlan"
    L = 1.0
    step_size = 0.1
    num_steps = 4

    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    sampling_alg = blackjax.mclmc(
        model.logdensity_fn,
        L=L,
        step_size=step_size,
        integrator=integrator,
    )

    state_transform = lambda x: x.position

    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda state, info: state_transform(state),
        progress_bar=True,
    )

    print("average of steps (slow way):", samples.mean(axis=0))

    memory_efficient_sampling_alg, transform = store_only_expectation_values(
        sampling_algorithm=sampling_alg, state_transform=state_transform
    )

    initial_state = memory_efficient_sampling_alg.init(initial_state)

    final_state, trace_at_every_step = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=memory_efficient_sampling_alg,
        num_steps=num_steps,
        transform=transform,
        progress_bar=True,
    )

    print("average of steps (fast way):", trace_at_every_step[0][-1])


def test_thinning():

    init_key, state_key, run_key = jax.random.split(jax.random.PRNGKey(0), 3)
    model = StandardNormal(2)
    initial_position = model.sample_init(init_key)
    initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
    )
    integrator_type = "mclachlan"
    L = 1.0
    step_size = 0.1
    num_steps = 1000

    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    sampling_alg = blackjax.mclmc(
        model.logdensity_fn,
        L=L,
        step_size=step_size,
        integrator=integrator,
    )

    sampling_alg = thinning_kernel(sampling_alg, thinning_factor=10)

    state_transform = lambda x: x.position

    _, samples = run_inference_algorithm(
        rng_key=run_key,
        initial_state=initial_state,
        inference_algorithm=sampling_alg,
        num_steps=num_steps,
        transform=lambda state, info: (state_transform(state)),
        progress_bar=True,
    )

    print(jnp.var(samples, axis=0))
    # print(samples)


# this function simply runs all of the samplers and some of the models, to make sure that everything is working
def test_benchmarking():

    model = StandardNormal(10)
    # model = Brownian()
    integrator_type = "mclachlan"
    num_steps = 2000
    num_chains = 128
    key1 = jax.random.PRNGKey(1)

    init_key, state_key, run_key = jax.random.split(key1, 3)
    initial_position = model.sample_init(init_key)
    unadjusted_initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
    )
    adjusted_initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=model.logdensity_fn,
        random_generator_arg=state_key,
    )

    if False:
        ess, ess_avg, ess_corr, _, acceptance_rate = benchmark_chains(
            model,
            run_nuts(integrator_type="velocity_verlet", preconditioning=False),
            key1,
            n=num_steps,
            batch=num_chains,
        )

        print(f"Effective Sample Size (ESS) of NUTS with preconditioning set to {False} is {ess_avg}")

        ess, ess_avg, ess_corr, _, acceptance_rate = benchmark_chains(
            model,
            run_unadjusted_mclmc_no_tuning(
                # L=0.2,
                # step_size=5.34853,
                step_size=3.56,
                L=2.642,
                integrator_type='velocity_verlet',
                initial_state=unadjusted_initial_state,
                sqrt_diag_cov=1.0,
            ),
            run_key,
            n=num_steps,
            batch=num_chains,
        )

        print(f"Effective Sample Size (ESS) of untuned unadjusted mclmc with preconditioning set to {False} is {ess_avg}")

        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
            model,
            run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=False),
            key1,
            n=num_steps,
            batch=num_chains,
        )
        print(f"Effective Sample Size (ESS) of tuned unadjusted mclmc with preconditioning set to {False} is {ess_avg}")

        ess, ess_avg, ess_corr, _, acceptance_rate = benchmark_chains(
            model,
            run_adjusted_mclmc_no_tuning(
                integrator_type=integrator_type,
                # step_size=3.9834502 ,
                # L=4.3817954,
                step_size=0.4,
                L=2.3,
                sqrt_diag_cov=1.0,
                initial_state=adjusted_initial_state,
            ),
            key1,
            n=num_steps,
            batch=num_chains,
        )
        print(f"Effective Sample Size (ESS) of untuned adjusted mclmc with preconditioning set to {False} is {ess_avg}")

    if True:
        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
            model,
            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.0, target_acc_rate=0.9),
            key1,
            n=num_steps,
            batch=num_chains,
        )
        print(f"Effective Sample Size (ESS) of tuned adjusted mclmc with preconditioning set to {False} is {ess_avg}")
        
        
        ess, ess_avg, ess_corr, params, acceptance_rate = benchmark_chains(
            model,
            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.1, target_acc_rate=0.9),
            key1,
            n=num_steps,
            batch=num_chains,
        )
        print(f"Effective Sample Size (ESS) of tuned adjusted mclmc (stage 3) with preconditioning set to {False} is {ess_avg}")

    ## grid search
        
    if False:

        grid_key, bench_key = jax.random.split(key1, 2)

        def func(L, step_size):
                
                return 10, 1

                ess, ess_avg, ess_corr, params, acceptance_rate  = benchmark_chains(
                    model=model,
                    sampler=run_adjusted_mclmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=adjusted_initial_state,
                        sqrt_diag_cov=1.,
                        L=L,
                        step_size=step_size,
                        L_proposal_factor=jnp.inf,
                    ),
                    key=grid_key,
                    n=num_steps,
                    batch=3,
                )

                jax.debug.print("x {x}", x=r[3].L.shape)

                return ess, (params.L, params.step_size)

        results, edge = grid_search(
            func=func,
            x=3.0,
            y=1.0,
            delta_x=3.0 - 0.2,
            delta_y=1.0 - 0.2,
            size_grid=3,
            num_iter=2,
        )

        print(results)


if __name__ == "__main__":

    # todo:
    # grid search test

    test_benchmarking()
    # benchmark_mhmchmc(batch_size=128)

    # test_thinning()

    # run_benchmarks_simple()


    # benchmark_omelyan(1)

    # try_new_run_inference()

    # print(grid_search(func, 0., 0., 1., 2., size_grid= 5, num_iter=1))
    # run_benchmarks(128)
    # run_benchmarks_step_size(128)
    # run_benchmarks(128)
    # benchmark_omelyan(10)
    # print("4")
