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
from blackjax.adaptation.adjusted_mclmc_adaptation import adjusted_mclmc_make_L_step_size_adaptation

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
    Rosenbrock,
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
    IllConditionedGaussian(100, k): {'mclmc': 20000, 'adjusted_mclmc': 20000, 'nuts': 20000}
    for k in np.ceil(np.logspace(1, 5, num=10)).astype(int)
}

# models = {
#     StandardNormal(10) : {'mclmc': 2000, 'adjusted_mclmc' : 2000, 'nuts': 2000},
#     StandardNormal(50) : {'mclmc': 2000, 'adjusted_mclmc' : 2000, 'nuts': 2000},
#     StandardNormal(100) : {'mclmc': 2000, 'adjusted_mclmc' : 2000, 'nuts': 2000},
#     StandardNormal(500) : {'mclmc': 4000, 'adjusted_mclmc' : 4000, 'nuts': 4000},
#     StandardNormal(1000) : {'mclmc': 4000, 'adjusted_mclmc' : 4000, 'nuts': 4000},
#     Brownian(): {"mclmc": 10000, "adjusted_mclmc": 10000, "nuts": 10000},
#     GermanCredit(): {'mclmc': 80000, 'adjusted_mclmc' : 80000, 'nuts': 80000},
#     ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
#     Rosenbrock(): {'mclmc': 20000, 'adjusted_mclmc' : 20000, 'nuts': 20000},
# }

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

def grid_search(func, x, y, delta_x, delta_y, grid_size=5, num_iter=3):
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
        Z = np.linspace(z - delta_z, z + delta_z, grid_size)
        # jax.debug.print("grid {x}", x=Z)
        Results = [[func(xx, yy) for yy in Z[:, 1]] for xx in Z[:, 0]]
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
        state, results, edge = kernel(state)
        jax.debug.print("optimal result on iteration {x}", x=(iteration, results[0]))
        # jax.debug.print("Optimal params on iteration: {x}", x=(results[1]))
        # jax.debug.print("Optimal score on iteration: {x}", x=(results[0]))
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

    params, grad_calls_per_traj, acceptance_rate, expectation, ess_corr = pvmap(
        lambda pos, key: sampler(
            model=model, num_steps=n, initial_position=pos, key=key
        )
    )(init_pos, keys)
    avg_grad_calls_per_traj = jnp.nanmean(grad_calls_per_traj, axis=0)

    err_t_median_avg = jnp.median(expectation[:, :, 0], axis=0)
    esses_avg, grads_to_low_avg, _ = calculate_ess(
        err_t_median_avg, grad_evals_per_step=avg_grad_calls_per_traj
    )

    err_t_median_max = jnp.median(expectation[:, :, 1], axis=0)
    esses_max, _, _ = calculate_ess(
        err_t_median_max, grad_evals_per_step=avg_grad_calls_per_traj
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
    return esses_max.item(), esses_avg.item(), ess_corr, params, jnp.mean(acceptance_rate, axis=0), grads_to_low_avg



def benchmark_adjusted_mclmc(batch_size, key_index=1):

    key0, key1 = jax.random.split(jax.random.PRNGKey(key_index), 2)


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

                        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
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
                        x=blackjax_adjusted_mclmc_sampler_params.L*2,
                        y=blackjax_adjusted_mclmc_sampler_params.step_size*2,
                        # x=3.316967,
                        # y=0.390205,
                        delta_x=blackjax_adjusted_mclmc_sampler_params.L*2 - 0.2,
                        delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*2 - 0.2,
                        grid_size=6,
                        num_iter=3,
                    )


                    print("BENCHMARK after finding optimal params with grid \n\n\n")
                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                        model,
                        run_adjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_state_after_tuning,
                            return_ess_corr=True,
                        ),
                        bench_key,
                        n=num_steps,
                        batch=num_chains,
                    )
                    print(f"best from grid search adjusted {ess}")

                    # jax.debug.print("\nESS is {x}", x=(ess, ess_avg))
                    # jax.debug.print("\nESS CORR is {x}", x=ess_corr)
                    # jax.debug.print("{x} \n acceptance rate", x=acceptance_rate)

                    # # new_L = 0.4 * out[0] / (ess_corr * acceptance_rate)
                    # new_L = 0.4 * out[0] * acceptance_rate / (ess_corr)
                    # jax.debug.print("{x} L old/new", x=(out[0], new_L))


                    # def func_L(L, step_size):

                    #     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                    #         model=model,
                    #         sampler=run_adjusted_mclmc_no_tuning(
                    #             integrator_type=integrator_type,
                    #             initial_state=blackjax_state_after_tuning,
                    #             sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                    #             L=new_L,
                    #             step_size=step_size,
                    #             L_proposal_factor=L_proposal_factor,
                    #         ),
                    #         key=grid_key,
                    #         n=num_steps,
                    #         batch=batch_size,
                    #     )

                    #     return ess, (params.L.mean(), params.step_size.mean())
                    
                    # out_new, edge = grid_search(
                    #     func=func_L,
                    #     x=blackjax_adjusted_mclmc_sampler_params.L*2,
                    #     y=blackjax_adjusted_mclmc_sampler_params.step_size*2,
                    #     # x=3.316967,
                    #     # y=0.390205,
                    #     delta_x=blackjax_adjusted_mclmc_sampler_params.L*2 - 0.2,
                    #     delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*2 - 0.2,
                    #     grid_size=5,
                    #     num_iter=3,
                    # )

                    # jax.debug.print("{x} stepsize old/new", x=(out[1], out_new[1]))

                    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                    #     model,
                    #     run_adjusted_mclmc_no_tuning(
                    #         integrator_type=integrator_type,
                    #         step_size=out_new[1],
                    #         L=new_L,
                    #         sqrt_diag_cov=1.0,
                    #         initial_state=blackjax_state_after_tuning,
                    #     ),
                    #     bench_key,
                    #     n=num_steps,
                    #     batch=num_chains,
                    # )
                    # jax.debug.print("{x} ESS on 3-perfect", x=(ess, ess_avg))
                    # raise Exception
                   

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
                            ess_corr.mean().item(),
                            ess_corr.min().item(),
                            (1/(1/ess_corr).mean()).item(),
                            num_steps,
                            True,
                            1
                        )
                    ] = ess

                

            
            
            ####### run mclmc with standard tuning
            for preconditioning in [False, True]:

                unadjusted_with_tuning_key, adjusted_with_tuning_key, adjusted_with_tuning_key_stage3, nuts_key_with_tuning = jax.random.split(key0, 4)


                if True:
                    
                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                        model,
                        run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning),
                        unadjusted_with_tuning_key,
                        n=num_steps,
                        batch=num_chains,
                    )

                    
                    results[
                        (
                            model.name, model.ndims, "mclmc", params.L.mean().item(), params.step_size.mean().item(), (integrator_type), "standard", 1.0, preconditioning, 0, ess_avg, ess_corr.mean().item(), ess_corr.min().item(), (1/(1/ess_corr).mean()).item(), num_steps, False, 1
                        )
                    ] = ess
                    print(f"unadjusted mclmc with tuning, grads to low bias avg {grads_to_low_avg}")

                    ####### run adjusted_mclmc with standard tuning
                    for target_acc_rate, L_proposal_factor, max, num_windows in itertools.product(
                        [0.9], [jnp.inf], [True, False], [1,2,3]
                    ):  # , 3., 1.25, 0.5] ):
                        # coeffs = mclachlan_coefficients
                        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                            model,
                            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune3=0.0, L_proposal_factor=L_proposal_factor,
                            target_acc_rate=target_acc_rate, return_ess_corr=True, max=max, num_windows=num_windows),
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
                                ess_corr.mean().item(),
                                ess_corr.min().item(), (1/(1/ess_corr).mean()).item(),
                                num_steps,
                                max,
                                num_windows
                            )
                        ] = ess

                        

                        # print(f"adjusted_mclmc with tuning grads to low bias avg {grads_to_low_avg}")

                        # # integrator_type = mclachlan_coefficients
                        # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                        #     model,
                        #     run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune3=0.1, L_proposal_factor=L_proposal_factor,
                        #     target_acc_rate=target_acc_rate,return_ess_corr=True),
                        #     adjusted_with_tuning_key_stage3,
                        #     n=num_steps,
                        #     batch=num_chains,
                        # )
                        # results[
                        #     (
                        #         model.name,
                        #         model.ndims,
                        #         "mhmclmc:st3:" + str(target_acc_rate),
                        #         jnp.nanmean(params.L).item(),
                        #         jnp.nanmean(params.step_size).item(),
                        #         (integrator_type),
                        #         "standard",
                        #         acceptance_rate.mean().item(),
                        #         preconditioning,
                        #         1 / L_proposal_factor,
                        #         ess_avg,
                        #         ess_corr.mean().item(),
                        #         ess_corr.min().item(), (1/(1/ess_corr).mean()).item(),
                        #         num_steps,
                        #         False
                        #     )
                        # ] = ess
                        # print(f"adjusted_mclmc with stage 3 tuning, grads to low avg {grads_to_low_avg}")


                
                ####### run nuts
                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                    model,
                    run_nuts(integrator_type="velocity_verlet", preconditioning=preconditioning),
                    nuts_key_with_tuning,
                    n=num_steps,
                    batch=num_chains,
                )
                print(f"nuts, grads to low avg {grads_to_low_avg}")
                
                results[
                    (
                        model.name,
                        model.ndims,
                        "nuts",
                        0.0,
                        0.0,
                        "velocity_verlet",
                        "standard",
                        acceptance_rate.mean().item(),
                        preconditioning,
                        0,
                        ess_avg,
                        ess_corr.mean().item(),
                        ess_corr.min().item(), (1/(1/ess_corr).mean()).item(),
                        num_steps,
                        None,
                        -1,
                    )
                ] = ess

        df = pd.Series(results).reset_index()
        df.columns = [
            "model", "dims", "sampler", "L", "step_size", "integrator", "tuning", "acc_rate", "preconditioning", "inv_L_prop", "ess_avg", "ess_corr_avg", "ess_corr_min", "ess_corr_inv_mean", "num_steps", "worst", "num_windows", "ESS"]
        # df.result = df.result.apply(lambda x: x[0].item())
        # df.model = df.model.apply(lambda x: x[1])
        df.to_csv(f"results{model.name}{model.ndims}{key_index}.csv", index=False)

    return results


# TODO: not updated to new code yet!
def benchmark_omelyan(batch_size):

    key = jax.random.PRNGKey(2)
    results = defaultdict(tuple)
    for variables in itertools.product(
        # ["adjusted_mclmc", "nuts", "mclmc", ],
        [
            StandardNormal(d)
            for d in np.ceil(np.logspace(np.log10(1e1), np.log10(1e6), 20)).astype(int)
        ],
        # [
        #     StandardNormal(10)
            
        # ],
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

        num_steps = 2000

        blackjax_state_after_tuning, blackjax_adjusted_mclmc_sampler_params = adjusted_mclmc_tuning(
            initial_position=model.sample_init(init_pos_key),
            num_steps=num_steps,
            rng_key=tune_key,
            logdensity_fn=model.logdensity_fn,
            integrator_type=integrator_type,
            diagonal_preconditioning=False,
            target_acc_rate=target_acceptance_rate_of_order[integrator_order(integrator_type)],
        )

        print(
            f"\nModel: {model.name,model.ndims}, Sampler: Adjusted MCLMC\n Integrator: {integrator_type}\nNumber of chains {num_chains}",
        )
        print(
            f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
        )

        def func(L, step_size):

            ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                model=model,
                sampler=run_adjusted_mclmc_no_tuning(
                    integrator_type=integrator_type,
                    initial_state=blackjax_state_after_tuning,
                    sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                    L=L,
                    step_size=step_size,
                    L_proposal_factor=jnp.inf,
                ),
                key=grid_key,
                n=num_steps,
                batch=batch_size,
            )

            return ess_avg, (params.L.mean(), params.step_size.mean())

        out, edge = grid_search(
            func=func,
            x=blackjax_adjusted_mclmc_sampler_params.L*2,
            y=blackjax_adjusted_mclmc_sampler_params.step_size*2,
            delta_x=blackjax_adjusted_mclmc_sampler_params.L*2 - 0.2,
            delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*2 - 0.2,
            grid_size=6,
            num_iter=5,
        )

        # results[(model.name, model.ndims, "mhmchmc:grid_new", out[0].item(), out[1].item(), integrator_type, f"gridsearch", out[3].item(), True, 1/L_proposal_factor, "n/a", "n/a", num_steps)] = out[2].item()

        print("BENCHMARK after finding optimal params with grid \n\n\n")
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
            model,
            run_adjusted_mclmc_no_tuning(
                integrator_type=integrator_type,
                initial_state=blackjax_state_after_tuning,
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

        jax.debug.print("x {x}", x=(ess_avg, ess, out[0], out[1]))
        results[
            (
                model.name,
                model.ndims,
                "mclmc",
                (integrator_type),
                acceptance_rate.item(),
                edge,
                out[0].item(),
                out[1].item(),
            )
        ] = ess_avg

        # raise Exception

    save = True
    if save:

        df = pd.Series(results).reset_index()
        df.columns = [
            "model",
            "dims",
            "sampler",
            "integrator",
            "acc_rate",
            "convergence",
            "L",
            "step_size",
            "ESS AVG",
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

    # model = StandardNormal(1000)
    model = IllConditionedGaussian(100, 12916)
    # model = Brownian()
    integrator_type = "mclachlan"
    num_steps = 20000
    num_chains = 12

    preconditioning = False

    key1 = jax.random.PRNGKey(2)

    init_key, state_key, tune_key, run_key = jax.random.split(key1, 4)
    initial_position = model.sample_init(init_key)
    unadjusted_initial_state = blackjax.mcmc.mclmc.init(
        position=initial_position, logdensity_fn=model.logdensity_fn, rng_key=state_key
    )
    adjusted_initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=model.logdensity_fn,
        random_generator_arg=state_key,
    )



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
        diagonal_preconditioning=preconditioning,
    )

    # ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
    #     model,
    #     run_nuts(integrator_type="velocity_verlet", preconditioning=preconditioning),
    #     run_key,
    #     n=num_steps,
    #     batch=num_chains,
    # )
    # print(f"Effective Sample Size (ESS) of NUTS with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}")

    if True:
        # print(f"acc rate is {acceptance_rate}")


        # ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
        #     model,
        #     run_unadjusted_mclmc_no_tuning(
        #         # L=0.2,
        #         # step_size=5.34853,
        #         step_size=3.56,
        #         L=1.888073,
        #         integrator_type='velocity_verlet',
        #         initial_state=unadjusted_initial_state,
        #         sqrt_diag_cov=1.0,
        #         return_ess_corr=True
        #     ),
        #     run_key,
        #     n=num_steps,
        #     batch=num_chains,
        # )

        # print(f"Effective Sample Size (ESS) of untuned unadjusted mclmc with preconditioning set to {False} is {ess_avg}")

        # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
        #     model,
        #     run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning),
        #     run_key,
        #     n=num_steps,
        #     batch=num_chains,
        # )
        # print(f"Effective Sample Size (ESS) of tuned unadjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max{ess}")



        # ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
        #     model,
        #     run_adjusted_mclmc_no_tuning(
        #         integrator_type=integrator_type,
        #         step_size=0.400800,
        #         L=1.888073,
        #         # step_size=4.61,
        #         # L=4.670475,
        #         sqrt_diag_cov=1.0,
        #         initial_state=blackjax_state_after_tuning,
        #         return_ess_corr=False
        #     ),
        #     run_key,
        #     n=num_steps,
        #     batch=num_chains,
        # )
        # print(f"Effective Sample Size (ESS) of untuned adjusted mclmc with preconditioning set to {False} is avg {ess_avg} and max {ess} with acc rate of {acceptance_rate}")



   
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
            model,
            run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.0, target_acc_rate=0.9, return_ess_corr=True, max=False, num_windows=2),
            run_key,
            n=num_steps,
            batch=num_chains,
        )
        print(f"Effective Sample Size (ESS) of tuned adjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}, with L {params.L.mean()} and stepsize {params.step_size.mean()}")
        
        
        # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
        #     model,
        #     run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.5, target_acc_rate=0.9, return_ess_corr=True),
        #     run_key,
        #     n=num_steps,
        #     batch=num_chains,
        # )
        # print(f"Effective Sample Size (ESS) of tuned adjusted mclmc (stage 3) with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}, with L {params.L.mean()} and stepsize {params.step_size.mean()}")
        print(f"acc rate is {acceptance_rate}")
        print(f"ess corr is {ess_corr.min()}")
        print(f"stage 2 L (max=False) is {params.L.mean()}")
        print(f"optimal L avg is {np.sqrt(np.mean(model.E_x2)*model.ndims)}")
        print(f"optimal L max is {np.sqrt(np.max(model.E_x2)*model.ndims)}")
        # print(f"L/esscorr is {params.L.mean()/ess_corr.mean()}")
        # print(f'prefactor is {(ess_corr.mean())}')
        
        

    ## grid search
        
    if False:

        grid_key, bench_key, tune_key, init_key = jax.random.split(run_key, 4)

        initial_position = model.sample_init(init_key)

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

        def func(L, step_size):
                

                ess, ess_avg, ess_corr, params, acceptance_rate  = benchmark_chains(
                    model=model,
                    sampler=run_adjusted_mclmc_no_tuning(
                        integrator_type=integrator_type,
                        initial_state=blackjax_state_after_tuning,
                        sqrt_diag_cov=1.,
                        L=L,
                        step_size=step_size,
                        L_proposal_factor=jnp.inf,
                    ),
                    key=grid_key,
                    n=num_steps,
                    batch=128,
                )


                return ess_avg, (params.L, params.step_size)
        

        out, edge = grid_search(
                        func=func,
                        x=blackjax_adjusted_mclmc_sampler_params.L*2,
                        y=blackjax_adjusted_mclmc_sampler_params.step_size*2,
                        # x=3.316967,
                        # y=0.390205,
                        delta_x=blackjax_adjusted_mclmc_sampler_params.L*2 - 0.2,
                        delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*2 - 0.2,
                        grid_size=6,
                        num_iter=4,
                    )
        
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
                        model,
                        run_adjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_state_after_tuning,
                            return_ess_corr=False,
                        ),
                        bench_key,
                        n=num_steps,
                        batch=num_chains,
                    )
        
        jax.debug.print("ess, ess_avg {x}", x=(ess, ess_avg))


def grid_search_only_L(model, num_steps, num_chains, target_acc_rate, integrator_type, key, grid_size, z, delta_z, state, opt='max'):

    Lgrid = np.linspace(z - delta_z, z + delta_z, grid_size)
    # Lgrid = np.array([z])
    ESS = np.zeros_like(Lgrid)
    ESS_AVG = np.zeros_like(Lgrid)
    ESS_CORR_AVG = np.zeros_like(Lgrid)
    ESS_CORR_MAX = np.zeros_like(Lgrid)
    STEP_SIZE = np.zeros_like(Lgrid)
    RATE = np.zeros_like(Lgrid)
    integrator = map_integrator_type_to_integrator["mclmc"][integrator_type]

    da_key, bench_key = jax.random.split(key, 2)

    for i in range(len(Lgrid)):
        jax.debug.print("L {x}", x=(Lgrid[i]))

        params = MCLMCAdaptationState(
            L=Lgrid[i],
            step_size=Lgrid[i]/5,
            sqrt_diag_cov=1.0,
        )

        kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
            integrator=integrator,
            integration_steps_fn=lambda k: jnp.ceil(
                jax.random.uniform(k) * rescale(avg_num_integration_steps)
            ),
            sqrt_diag_cov=sqrt_diag_cov,
        )(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            logdensity_fn=model.logdensity_fn,
            L_proposal_factor=jnp.inf,
        )

        (
            blackjax_state_after_tuning,
            params,
            _,
            _,
        ) = adjusted_mclmc_make_L_step_size_adaptation(
            kernel=kernel,
            dim=model.ndims,
            frac_tune1=1.0,
            frac_tune2=0.0,
            target=target_acc_rate,
            diagonal_preconditioning=False,
            fix_L_first_da=True,
        )(
            state, params, num_steps, da_key
        )

        # raise Exception

        # jax.debug.print("DA {x}", x=(final_da))
        jax.debug.print("benchmarking with L and step size {x}", x=(Lgrid[i], params.step_size))
        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
            model,
            run_adjusted_mclmc_no_tuning(
                integrator_type=integrator_type,
                initial_state=blackjax_state_after_tuning,
                sqrt_diag_cov=1.0,
                L=Lgrid[i],
                step_size=params.step_size,
                L_proposal_factor=jnp.inf,
                return_ess_corr=True,
            ),
            bench_key,
            n=num_steps,
            batch=num_chains,
        )
        jax.debug.print("{x} comparison of acceptance", x=(acceptance_rate, target_acc_rate))
        ESS[i] = ess
        ESS_AVG[i] = ess_avg
        ESS_CORR_AVG[i] = ess_corr.mean().item()
        ESS_CORR_MAX[i] = ess_corr.max().item()
        STEP_SIZE[i] = params.step_size.mean().item()
        RATE[i] = acceptance_rate.mean().item()
    # iopt = np.argmax(ESS)
    if opt=='max':
        iopt = np.argmax(ESS)
    else:
        iopt = np.argmax(ESS_AVG)

    return Lgrid[iopt], STEP_SIZE[iopt], ESS[iopt], ESS_AVG[iopt], ESS_CORR_MAX[iopt], ESS_CORR_AVG[iopt], RATE[iopt]

def benchmark_ill_conditioned(batch_size=1,key_index=2):

    # How about we test these things numerically? Pick say d = 100 and determine the optimal L with grid search for a range of Ill Conditioned Gaussians (say kappa \in np.logspace(0, 5) ). But do this at fixed acceptance rate, not optimal stepsize (use dual averaging), say at a \in  {0.1, 0.3, 0.6, 0.9}. For each (L-optimal) chain also determine tauint. We can then plot these relations.

    ndims = 100
    num_chains = batch_size
    num_steps = 10000
    results = defaultdict(tuple)
    integrator_type = "mclachlan"

    tune_key, init_pos_key, grid_key = jax.random.split(jax.random.PRNGKey(key_index), 3)

    # for acc_rate in [0.1, 0.3, 0.6, 0.9]:
    # for acc_rate in np.linspace(0.1, 0.9, 20):
    for acc_rate in [0.8]:
        # for kappa in np.logspace(0, 5, 5):
        for kappa in np.ceil(np.logspace(np.log10(10), np.log10(100000), 10)).astype(int):
            model = IllConditionedGaussian(ndims, kappa)
            # model = Brownian()
            # model = StandardNormal(ndims)
            print(f"Model: {model.name}, kappa: {kappa}, acc_rate: {acc_rate}")

            initial_position = model.sample_init(init_pos_key)

            (
                state,
                blackjax_params_after_tuning,
            ) = adjusted_mclmc_tuning(
                initial_position=initial_position,
                num_steps=num_steps,
                rng_key=tune_key,
                logdensity_fn=model.logdensity_fn,
                integrator_type=integrator_type,
                frac_tune2=0.1,
                frac_tune3=0.0,
                target_acc_rate=acc_rate,
                diagonal_preconditioning=False,
            )


            L, stepsize, ess, ess_avg, ess_corr_max, ess_corr_avg, rate  =grid_search_only_L(model, num_steps, num_chains, acc_rate, integrator_type, grid_key, 40, blackjax_params_after_tuning.L*2, blackjax_params_after_tuning.L*2-0.2, state)



            results[
                (
                    model.name,
                    model.ndims,
                    kappa,
                    acc_rate,
                    rate,
                    "mclmc",
                    integrator_type,
                    L,
                    stepsize,
                    ess,
                    ess_avg,
                    ess_corr_max,
                )
            ] = ess_corr_avg

    save = True
    print(results)
    if save:

        df = pd.Series(results).reset_index()
        df.columns = [
            "model",
            "dims",
            "kappa",
            "target_acc_rate",
            "true_rate",
            "sampler",
            "integrator",
            "L",
            "step_size",
            "ess",
            "ess_avg",
            "ess_corr_max",
            "ess_corr_avg",
        ]
        # df.result = df.result.apply(lambda x: x[0].item())
        # df.model = df.model.apply(lambda x: x[1])
        df.to_csv("grid_search_L"+str(key_index)+".csv", index=False)


def test_da_functionality():

    model = StandardNormal(100)
    # model = Brownian()
    num_steps = 10000
    num_chains = 128
    key = jax.random.PRNGKey(20)
    
    params = MCLMCAdaptationState(
        L=jnp.sqrt(model.ndims)*5, step_size=jnp.sqrt(model.ndims), sqrt_diag_cov=1.0,
    )

    init_key, state_key, run_key, tune_key = jax.random.split(key, 4)
    initial_position = model.sample_init(init_key)
    adjusted_initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=model.logdensity_fn,
        random_generator_arg=state_key,
    )

    kernel = lambda rng_key, state, avg_num_integration_steps, step_size, sqrt_diag_cov: blackjax.mcmc.adjusted_mclmc.build_kernel(
        integrator=map_integrator_type_to_integrator["mclmc"]["mclachlan"],
        integration_steps_fn=lambda k: jnp.ceil(
            jax.random.uniform(k) * rescale(avg_num_integration_steps)
        ),
        sqrt_diag_cov=sqrt_diag_cov,
    )(
        rng_key=rng_key,
        state=state,
        step_size=step_size,
        logdensity_fn=model.logdensity_fn,
        L_proposal_factor=jnp.inf,
    )

    (
        blackjax_state_after_tuning,
        params,
        _,
        _,
    ) = adjusted_mclmc_make_L_step_size_adaptation(
        kernel=kernel,
        dim=model.ndims,
        frac_tune1=0.1,
        frac_tune2=0.0,
        target=0.9,
        diagonal_preconditioning=False,
        fix_L_first_da=True,
    )(
        adjusted_initial_state, params, num_steps, tune_key
    )

    raise Exception

    
    
    # ### second DA
    # (
    #     blackjax_state_after_tuning,
    #     params,
    #     _,
    #     _,
    # ) = adjusted_mclmc_make_L_step_size_adaptation(
    #     kernel=kernel,
    #     dim=model.ndims,
    #     frac_tune1=0.1,
    #     frac_tune2=0.0,
    #     target=0.8,
    #     diagonal_preconditioning=False,
    #     fix_L_first_da=True,
    # )(
    #     blackjax_state_after_tuning, params, num_steps, key
    # )

    # blackjax_state_after_tuning, params = adjusted_mclmc_tuning(
    #     initial_position=initial_position,
    #     num_steps=num_steps,
    #     rng_key=tune_key,
    #     logdensity_fn=model.logdensity_fn,
    #     integrator_type="mclachlan",
    #     frac_tune3=0.0,
    #     target_acc_rate=0.9,
    #     diagonal_preconditioning=False,
    # )

    jax.debug.print("params step size {x}", x=params.step_size)
    jax.debug.print("params L {x}", x=params.L)

    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
        model,
        run_adjusted_mclmc_no_tuning(
            integrator_type="mclachlan",
            initial_state=blackjax_state_after_tuning,
            # initial_state=adjusted_initial_state,
            sqrt_diag_cov=1.0,
            L=params.L,
            step_size=params.step_size,
            L_proposal_factor=jnp.inf,
        ),
        run_key,
        n=num_steps,
        batch=num_chains,
    )

    print(acceptance_rate, ess)

if __name__ == "__main__":



    # todo:
    # grid search test


    # test_da_functionality()

    test_benchmarking()
    # benchmark_ill_conditioned(batch_size=10)
    # for i in range(1,10):
    # benchmark_adjusted_mclmc(batch_size=10, key_index=20)

    # benchmark_ill_conditioned(batch_size=128)

    # test_thinning()



    # benchmark_omelyan(4)

    # try_new_run_inference()

    # print(grid_search(func, 0., 0., 1., 2., grid_size= 5, num_iter=1))
    # run_benchmarks(128)
    # run_benchmarks_step_size(128)
    # run_benchmarks(128)
    # benchmark_omelyan(10)
    # print("4")
