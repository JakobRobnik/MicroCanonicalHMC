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
from metrics import benchmark_chains, grid_search

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
    Gaussian,
    GermanCredit,
    ItemResponseTheory,
    MixedLogit,
    Rosenbrock,
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


def benchmark(batch_size, models, key_index=1, do_grid_search=True, do_non_grid_search=True, integrators = ["mclachlan"]):

    keys_for_not_grid, keys_for_grid = jax.random.split(jax.random.PRNGKey(key_index), 2)

    do_grid_search_for_adjusted = True and do_grid_search
    do_grid_search_for_unadjusted = False and do_grid_search
    do_unadjusted_mclmc = True
    do_nuts = True

    num_chains = batch_size  # 1 + batch_size//model.ndims
    
    for model in models:
        print(f"Running benchmark for {model.name} with {model.ndims} dimensions")
        results = defaultdict(tuple)
        if do_grid_search:
            print(
                f"NUMBER OF CHAINS for {model.name} and adjusted_mclmc is {num_chains}"
            )
            for integrator_type in integrators:


                ####### run adjusted_mclmc with standard tuning + grid search

                (
                    init_pos_key,
                    tune_key_adjusted,
                    tune_key_unadjusted,
                    grid_key,
                    bench_key,
                    unadjusted_grid_key,
                    unadjusted_bench_key,
                ) = jax.random.split(keys_for_grid, 7)




                initial_position = model.sample_init(init_pos_key)

                (
                    blackjax_adjusted_state_after_tuning,
                    blackjax_adjusted_mclmc_sampler_params,
                ) = adjusted_mclmc_tuning(
                    initial_position=initial_position,
                    num_steps=models[model]["adjusted_mclmc"],
                    rng_key=tune_key_adjusted,
                    logdensity_fn=model.logdensity_fn,
                    integrator_type=integrator_type,
                    frac_tune3=0.0,
                    target_acc_rate=0.9,
                    diagonal_preconditioning=False,
                )

                (blackjax_unadjusted_state_after_tuning, blackjax_unadjusted_mclmc_sampler_params) = unadjusted_mclmc_tuning(
                    initial_position=initial_position,
                    num_steps=models[model]["mclmc"],
                    rng_key=tune_key_unadjusted,
                    logdensity_fn=model.logdensity_fn,
                    integrator_type=integrator_type,
                    diagonal_preconditioning=False,
                )

                if do_grid_search_for_adjusted:
                    print(
                        f"target acceptance rate {target_acceptance_rate_of_order[integrator_order(integrator_type)]}"
                    )
                    print(
                        f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
                    )

                    L_proposal_factor = jnp.inf
                    def func(L, step_size):

                        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                            model=model,
                            sampler=run_adjusted_mclmc_no_tuning(
                                integrator_type=integrator_type,
                                initial_state=blackjax_adjusted_state_after_tuning,
                                sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                                L=L,
                                step_size=step_size,
                                L_proposal_factor=L_proposal_factor,
                            ),
                            key=grid_key,
                            n=models[model]["adjusted_mclmc"],
                            batch=batch_size,
                        )

                        return ess, (params.L.mean(), params.step_size.mean())

                    out, edge = grid_search(
                        func=func,
                        x=blackjax_adjusted_mclmc_sampler_params.L*3,
                        y=blackjax_adjusted_mclmc_sampler_params.step_size*3,
                        # x=3.316967,
                        # y=0.390205,
                        delta_x=blackjax_adjusted_mclmc_sampler_params.L*3 - 1.0,
                        delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*3 - 1.0,
                        grid_size=6,
                        num_iter=3,
                    )


                    print("BENCHMARK after finding optimal params with grid \n\n\n")
                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                        model,
                        run_adjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_adjusted_state_after_tuning,
                            return_ess_corr=True,
                        ),
                        bench_key,
                        n=models[model]["adjusted_mclmc"],
                        batch=num_chains,
                    )
                    print(f"best from grid search adjusted {ess}")

                   
                   

                    results[
                        (
                            model.name,
                            model.ndims,
                            f"adjusted_mclmc:grid_edge{edge}",
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
                            models[model]["adjusted_mclmc"],
                            num_chains,
                            True,
                            1
                        )
                    ] = ess
                    
                    
                    
                    # print("and now with ess_avg targeted")
                    # grid_key = jax.random.split(grid_key, 1)[0]
                    # bench_key = jax.random.split(bench_key, 1)[0]
                    # print(
                    #     f"target acceptance rate {target_acceptance_rate_of_order[integrator_order(integrator_type)]}"
                    # )
                    # print(
                    #     f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
                    # )

                    # L_proposal_factor = jnp.inf
                    # def func(L, step_size):

                    #     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                    #         model=model,
                    #         sampler=run_adjusted_mclmc_no_tuning(
                    #             integrator_type=integrator_type,
                    #             initial_state=blackjax_adjusted_state_after_tuning,
                    #             sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                    #             L=L,
                    #             step_size=step_size,
                    #             L_proposal_factor=L_proposal_factor,
                    #         ),
                    #         key=grid_key,
                    #         n=models[model]["adjusted_mclmc"],
                    #         batch=batch_size,
                    #     )

                    #     return ess_avg, (params.L.mean(), params.step_size.mean())

                    # out, edge = grid_search(
                    #     func=func,
                    #     x=blackjax_adjusted_mclmc_sampler_params.L*3,
                    #     y=blackjax_adjusted_mclmc_sampler_params.step_size*3,
                    #     # x=3.316967,
                    #     # y=0.390205,
                    #     delta_x=blackjax_adjusted_mclmc_sampler_params.L*3 - 1.0,
                    #     delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*3 - 1.0,
                    #     grid_size=6,
                    #     num_iter=3,
                    # )


                    # print("BENCHMARK after finding optimal params with grid \n\n\n")
                    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                    #     model,
                    #     run_adjusted_mclmc_no_tuning(
                    #         integrator_type=integrator_type,
                    #         step_size=out[1],
                    #         L=out[0],
                    #         sqrt_diag_cov=1.0,
                    #         initial_state=blackjax_adjusted_state_after_tuning,
                    #         return_ess_corr=True,
                    #     ),
                    #     bench_key,
                    #     n=models[model]["adjusted_mclmc"],
                    #     batch=num_chains,
                    # )
                    # print(f"best from grid search adjusted {ess}")

                   
                   

                    # results[
                    #     (
                    #         model.name,
                    #         model.ndims,
                    #         f"adjusted_mclmc:grid_edge{edge}",
                    #         jnp.nanmean(params.L).item(),
                    #         jnp.nanmean(params.step_size).item(),
                    #         integrator_type,
                    #         f"gridsearch",
                    #         acceptance_rate.mean().item(),
                    #         False,
                    #         1 / L_proposal_factor,
                    #         ess_avg,
                    #         ess_corr.mean().item(),
                    #         ess_corr.min().item(),
                    #         (1/(1/ess_corr).mean()).item(),
                    #         models[model]["adjusted_mclmc"],
                    #         num_chains,
                    #         False,
                    #         1
                    #     )
                    # ] = ess
                    
                    print("and now with L_proposal=1.25 targeted")
                    grid_key = jax.random.split(grid_key, 1)[0]
                    bench_key = jax.random.split(bench_key, 1)[0]
                    print(
                        f"target acceptance rate {target_acceptance_rate_of_order[integrator_order(integrator_type)]}"
                    )
                    print(
                        f"params after initial tuning are L={blackjax_adjusted_mclmc_sampler_params.L}, step_size={blackjax_adjusted_mclmc_sampler_params.step_size}"
                    )

                    L_proposal_factor = 1.25
                    def func(L, step_size):

                        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                            model=model,
                            sampler=run_adjusted_mclmc_no_tuning(
                                integrator_type=integrator_type,
                                initial_state=blackjax_adjusted_state_after_tuning,
                                sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                                L=L,
                                step_size=step_size,
                                L_proposal_factor=L_proposal_factor,
                                random_trajectory_length=False,
                            ),
                            key=grid_key,
                            n=models[model]["adjusted_mclmc"],
                            batch=batch_size,
                        )

                        return ess, (params.L.mean(), params.step_size.mean())

                    out, edge = grid_search(
                        func=func,
                        x=blackjax_adjusted_mclmc_sampler_params.L*3,
                        y=blackjax_adjusted_mclmc_sampler_params.step_size*3,
                        # x=3.316967,
                        # y=0.390205,
                        delta_x=blackjax_adjusted_mclmc_sampler_params.L*3 - 1.0,
                        delta_y=blackjax_adjusted_mclmc_sampler_params.step_size*3 - 1.0,
                        grid_size=6,
                        num_iter=3,
                    )


                    print("BENCHMARK after finding optimal params with grid \n\n\n")
                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                        model,
                        run_adjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_adjusted_state_after_tuning,
                            return_ess_corr=True,
                        ),
                        bench_key,
                        n=models[model]["adjusted_mclmc"],
                        batch=num_chains,
                    )
                    print(f"best from grid search adjusted {ess}")

                   
                   

                    results[
                        (
                            model.name,
                            model.ndims,
                            f"adjusted_mclmc:grid_edge{edge}",
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
                            models[model]["adjusted_mclmc"],
                            num_chains,
                            False,
                            1
                        )
                    ] = ess


                
                
                do_grid_search_for_unadjusted = True
                if do_grid_search_for_unadjusted:
                    
                    print(
                        f"params after initial tuning are L={blackjax_unadjusted_mclmc_sampler_params.L}, step_size={blackjax_unadjusted_mclmc_sampler_params.step_size}"
                    )

                    L_proposal_factor = jnp.inf
                    def func(L, step_size):

                        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                            model=model,
                            sampler=run_unadjusted_mclmc_no_tuning(
                                integrator_type=integrator_type,
                                initial_state=blackjax_unadjusted_state_after_tuning,
                                sqrt_diag_cov=blackjax_unadjusted_mclmc_sampler_params.sqrt_diag_cov,
                                L=L,
                                step_size=step_size,
                            ),
                            key=unadjusted_grid_key,
                            n=models[model]["mclmc"],
                            batch=batch_size,
                        )

                        return ess, (params.L.mean(), params.step_size.mean())

                    out, edge = grid_search(
                        func=func,
                        x=blackjax_unadjusted_mclmc_sampler_params.L*3,
                        y=blackjax_unadjusted_mclmc_sampler_params.step_size*3,
                        # x=3.316967,
                        # y=0.390205,
                        delta_x=blackjax_unadjusted_mclmc_sampler_params.L*3 - 1.0,
                        delta_y=blackjax_unadjusted_mclmc_sampler_params.step_size*3 - 1.0,
                        grid_size=6,
                        num_iter=3,
                    )


                    print("BENCHMARK after finding optimal params with grid \n\n\n")
                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                        model,
                        run_unadjusted_mclmc_no_tuning(
                            integrator_type=integrator_type,
                            step_size=out[1],
                            L=out[0],
                            sqrt_diag_cov=1.0,
                            initial_state=blackjax_unadjusted_state_after_tuning,
                            return_ess_corr=True,
                        ),
                        unadjusted_bench_key,
                        n=models[model]["mclmc"],
                        batch=num_chains,
                    )
                    print(f"best from grid search adjusted {ess}")

                   
                   

                    results[
                        (
                            model.name,
                            model.ndims,
                            f"mclmc:grid_edge{edge}",
                            jnp.nanmean(params.L).item(),
                            jnp.nanmean(params.step_size).item(),
                            integrator_type,
                            f"gridsearch",
                            1.0,
                            False,
                            1 / L_proposal_factor,
                            ess_avg,
                            ess_corr.mean().item(),
                            ess_corr.min().item(),
                            (1/(1/ess_corr).mean()).item(),
                            models[model]["mclmc"],
                            num_chains,
                            True,
                            1
                        )
                    ] = ess

                

            
            ##### save grid results
            df = pd.Series(results).reset_index()
            df.columns = [
                "model", "dims", "sampler", "L", "step_size", "integrator", "tuning", "acc_rate", "preconditioning", "inv_L_prop", "ess_avg", "ess_corr_avg", "ess_corr_min", "ess_corr_inv_mean", "num_steps", "num_chains", "worst", "num_windows", "ESS"]
            # df.result = df.result.apply(lambda x: x[0].item())
            # df.model = df.model.apply(lambda x: x[1])
            df.to_csv(f"gridresults{model.name}{model.ndims}{key_index}.csv", index=False)
            results = defaultdict(tuple)

        if not do_non_grid_search:
            return None
        
        for integrator_type in integrators:




            

            # keys_for_not_grid = jax.random.split(keys_for_not_grid, 1)[0]

            unadjusted_with_tuning_key, adjusted_with_tuning_key, adjusted_with_tuning_key_stage3, nuts_key_with_tuning = jax.random.split(keys_for_not_grid, 4)


            if do_unadjusted_mclmc:
                
                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                    model,
                    run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=False),
                    unadjusted_with_tuning_key,
                    n=models[model]["mclmc"],
                    batch=num_chains,
                )

                
                results[
                    (
                        model.name, model.ndims, "mclmc:st3", params.L.mean().item(), params.step_size.mean().item(), (integrator_type), "standard", 1.0, False, 0, ess_avg, ess_corr.mean().item(), ess_corr.min().item(), (1/(1/ess_corr).mean()).item(), models[model]["mclmc"], num_chains, False, 1
                    )
                ] = ess
                print(f"unadjusted mclmc with tuning, grads to low bias avg {grads_to_low_avg}")
                
                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                    model,
                    run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.0),
                    unadjusted_with_tuning_key,
                    n=models[model]["mclmc"],
                    batch=num_chains,
                )

                
                results[
                    (
                        model.name, model.ndims, "mclmc:st2", params.L.mean().item(), params.step_size.mean().item(), (integrator_type), "standard", 1.0, False, 0, ess_avg, ess_corr.mean().item(), ess_corr.min().item(), (1/(1/ess_corr).mean()).item(), models[model]["mclmc"], num_chains, False, 1
                    )
                ] = ess
                print(f"unadjusted stage 2 mclmc with tuning, grads to low bias avg {grads_to_low_avg}")

                ####### run adjusted_mclmc with standard tuning
            for target_acc_rate, (L_proposal_factor, random_trajectory_length), max, num_windows in itertools.product(
                    [0.9], [(jnp.inf, True), (1.25, False)], [True, False], [1,2]
                ):  # , 3., 1.25, 0.5] ):
                    # coeffs = mclachlan_coefficients

                    print(f"running adjusted mclmc with target acceptance rate {target_acc_rate}, L_proposal_factor {L_proposal_factor}, max {max}, num_windows {num_windows}")

                    adjusted_with_tuning_key = jax.random.split(adjusted_with_tuning_key, 1)[0]

                    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                        model,
                        run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=False, frac_tune3=0.0, L_proposal_factor=L_proposal_factor,
                        target_acc_rate=target_acc_rate, return_ess_corr=True, max=max, num_windows=num_windows, random_trajectory_length=random_trajectory_length),
                        adjusted_with_tuning_key,
                        n=models[model]["adjusted_mclmc"],
                        batch=num_chains,
                        
                    )
                    print(f"ess {ess}, ess_corr avg {ess_corr.mean()}, ess_corr min {ess_corr.min()}, ess_corr inv mean {1/(1/ess_corr).mean()}")
                    results[
                        (
                            model.name,
                            model.ndims,
                            "adjusted_mclmc:" + str(target_acc_rate),
                            jnp.nanmean(params.L).item(),
                            jnp.nanmean(params.step_size).item(),
                            (integrator_type),
                            "standard",
                            acceptance_rate.mean().item(),
                            False,
                            1 / L_proposal_factor,
                            ess_avg,
                            ess_corr.mean().item(),
                            ess_corr.min().item(), (1/(1/ess_corr).mean()).item(),
                            models[model]["adjusted_mclmc"],
                            num_chains,
                            max,
                            num_windows
                        )
                    ] = ess

                  

        if do_nuts:

            for integrator_type in ["velocity_verlet", "mclachlan"]:
                nuts_key_with_tuning = jax.random.split(keys_for_not_grid, 1)[0]
                ####### run nuts
                ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
                    model,
                    run_nuts(integrator_type=integrator_type, preconditioning=False),
                    nuts_key_with_tuning,
                    n=models[model]["nuts"],
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
                        integrator_type,
                        "standard",
                        acceptance_rate.mean().item(),
                        False,
                        0,
                        ess_avg,
                        ess_corr.mean().item(),
                        ess_corr.min().item(), (1/(1/ess_corr).mean()).item(),
                        models[model]["nuts"],
                        num_chains,
                        None,
                        -1,
                    )
                ] = ess

        df = pd.Series(results).reset_index()
        df.columns = [
            "model", "dims", "sampler", "L", "step_size", "integrator", "tuning", "acc_rate", "preconditioning", "inv_L_prop", "ess_avg", "ess_corr_avg", "ess_corr_min", "ess_corr_inv_mean", "num_steps", "num_chains", "worst", "num_windows", "ESS"]
        # df.result = df.result.apply(lambda x: x[0].item())
        # df.model = df.model.apply(lambda x: x[1])
        df.to_csv(f"results{model.name}{model.ndims}{key_index}.csv", index=False)

    return results





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
def optimal_L_for_gaussian(dims, kappa):

    model = Gaussian(dims, kappa, 'log')
    # model = Brownian()
    integrator_type = "mclachlan"
    num_steps = 20000
    num_chains = 128

    preconditioning = False

    key1 = jax.random.PRNGKey(2)

    init_key, search_key, tune_key, run_key, initial_key = jax.random.split(key1, 5)
    initial_position = model.sample_init(init_key)



    L = np.sqrt(np.max(model.E_x2))*model.ndims
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
        params = MCLMCAdaptationState(L=L, step_size=L*0.2, sqrt_diag_cov=1.0)
    )


    #### gridsearch epsilon
    # print(f"tuned L is {blackjax_adjusted_mclmc_sampler_params.L} and tuned step size is {blackjax_adjusted_mclmc_sampler_params.step_size}")

    # key1, key2, key3 = jax.random.split(key1, 3)
    # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark_chains(
    #                         model=model,
    #                         sampler=run_adjusted_mclmc_no_tuning(
    #                             integrator_type=integrator_type,
    #                             initial_state=blackjax_state_after_tuning,
    #                             sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
    #                             L=blackjax_adjusted_mclmc_sampler_params.L,
    #                             step_size=blackjax_adjusted_mclmc_sampler_params.step_size,
    #                             L_proposal_factor=L_proposal_factor,
    #                         ),
    #                         key=key1,
    #                         n=num_steps,
    #                         batch=num_chains,
    #     )
    # print(f"Effective Sample Size (ESS) of adjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}")
    
    step_size, ESS, ESS_AVG, ESS_CORR_MAX, ESS_CORR_AVG, RATE = grid_search_only_stepsize(L=L,model=model, num_steps=num_steps, num_chains=num_chains, integrator_type=integrator_type, key=search_key, grid_size=20,grid_boundaries=(0.1,L-0.1), state=blackjax_state_after_tuning, opt='max')

    print(f"step size is {step_size}")
    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _,_ = benchmark_chains(
                            model=model,
                            sampler=run_adjusted_mclmc_no_tuning(
                                integrator_type=integrator_type,
                                initial_state=blackjax_state_after_tuning,
                                sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
                                L=L,
                                step_size=step_size,
                                L_proposal_factor=jnp.inf,
                            ),
                            key=run_key,
                            n=num_steps,
                            batch=num_chains,
        )
    
    print(f"Effective Sample Size (ESS) of adjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}")

    results = defaultdict(tuple)
    results[
                        (
                            model.name,
                            model.ndims,
                            f"optimal_L_grid_eps",
                            jnp.nanmean(params.L).item(),
                            jnp.nanmean(params.step_size).item(),
                            integrator_type,
                            f"gridsearch",
                            acceptance_rate.mean().item(),
                            False,
                            1 / jnp.inf,
                            ess_avg,
                            ess_corr.mean().item(),
                            ess_corr.min().item(),
                            (1/(1/ess_corr).mean()).item(),
                            num_steps,
                            num_chains,
                            False,
                            -1
                        )
                    ] = ess
    df = pd.Series(results).reset_index()
    df.columns = [
        "model", "dims", "sampler", "L", "step_size", "integrator", "tuning", "acc_rate", "preconditioning", "inv_L_prop", "ess_avg", "ess_corr_avg", "ess_corr_min", "ess_corr_inv_mean", "num_steps", "num_chains", "worst", "num_windows", "ESS"]
    
def test_benchmarking():

    model = Gaussian(10, 1, 'linear')
    # model = Brownian()
    integrator_type = "mclachlan"
    num_steps = 2000
    num_chains = 128

    preconditioning = False

    key1 = jax.random.PRNGKey(2)

    init_key, state_key, tune_key, run_key, initial_key = jax.random.split(key1, 5)
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
        params = MCLMCAdaptationState(L=np.sqrt(model.ndims), step_size=np.sqrt(model.ndims)*0.2, sqrt_diag_cov=1.0)
    )

    L_proposal_factor = jnp.inf

    


    # bayesian_optimization = False
    # if bayesian_optimization:

    #     import bayex

    #     # def f(L, stepsize):
    #     #     return L-stepsize
        
    #     def make_f(key):
    #         def f(L, stepsize):

    #                     ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
    #                         model=model,
    #                         sampler=run_adjusted_mclmc_no_tuning(
    #                             integrator_type=integrator_type,
    #                             initial_state=blackjax_state_after_tuning,
    #                             sqrt_diag_cov=blackjax_adjusted_mclmc_sampler_params.sqrt_diag_cov,
    #                             L=L,
    #                             step_size=stepsize,
    #                             L_proposal_factor=L_proposal_factor,
    #                         ),
    #                         key=key,
    #                         n=num_steps,
    #                         batch=num_chains,
    #                     )
    #                     jax.debug.print("{x} ESS", x=ess)
    #                     jax.debug.print("{x} L stepsize", x=(L, stepsize))
    #                     return ess
    #         return f 

    #     # domain = {'L': bayex.domain.Real(0.1, 4.0), 'stepsize': bayex.domain.Real(0.1, 0.7)}
    #     domain = {'L': bayex.domain.Real(0.1, 2*blackjax_adjusted_mclmc_sampler_params.L.item()), 'stepsize': bayex.domain.Real(0.1, 2*blackjax_adjusted_mclmc_sampler_params.step_size.item())}
    #     optimizer = bayex.Optimizer(
    #             domain=domain, 
    #             maximize=True, acq='PI')

    #     # Define some prior evaluations to initialise the GP.
    #     # params = {'L': [1.944672], 'stepsize': [0.353445]}
    #     params = {'L': [blackjax_adjusted_mclmc_sampler_params.L], 'stepsize': [blackjax_adjusted_mclmc_sampler_params.step_size]}
    #     ys = [make_f(initial_key)(x,y) for x in params['L'] for y in params['stepsize']]
    #     opt_state = optimizer.init(ys, params)


    #     # Sample new points using Jax PRNG approach.
    #     ori_key = jax.random.key(42)
    #     for step in range(50):
    #         print(step)
    #         key = jax.random.fold_in(ori_key, step)
    #         key1, key2 = jax.random.split(key)
    #         new_params = optimizer.sample(key1, opt_state)
    #         y_new = make_f(key2)(**new_params)
    #         opt_state = optimizer.fit(opt_state, y_new, new_params)
        
    #     print(opt_state.best_params)

    # # ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
    # #     model,
    # #     run_nuts(integrator_type="velocity_verlet", preconditioning=preconditioning),
    # #     run_key,
    # #     n=num_steps,
    # #     batch=num_chains,
    # # )
    # # print(f"Effective Sample Size (ESS) of NUTS with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}")

    # if True:
    #     # print(f"acc rate is {acceptance_rate}")


    #     # ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
    #     #     model,
    #     #     run_unadjusted_mclmc_no_tuning(
    #     #         # L=0.2,
    #     #         # step_size=5.34853,
    #     #         step_size=3.56,
    #     #         L=1.888073,
    #     #         integrator_type='velocity_verlet',
    #     #         initial_state=unadjusted_initial_state,
    #     #         sqrt_diag_cov=1.0,
    #     #         return_ess_corr=True
    #     #     ),
    #     #     run_key,
    #     #     n=num_steps,
    #     #     batch=num_chains,
    #     # )

    #     # print(f"Effective Sample Size (ESS) of untuned unadjusted mclmc with preconditioning set to {False} is {ess_avg}")

    #     # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
    #     #     model,
    #     #     run_unadjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning),
    #     #     run_key,
    #     #     n=num_steps,
    #     #     batch=num_chains,
    #     # )
    #     # print(f"Effective Sample Size (ESS) of tuned unadjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max{ess}")



    #     ess, ess_avg, ess_corr, _, acceptance_rate, grads_to_low_avg = benchmark_chains(
    #         model,
    #         run_adjusted_mclmc_no_tuning(
    #             integrator_type=integrator_type,
    #             # step_size=opt_state.best_params['stepsize'],
    #             # L=opt_state.best_params['L'],
    #             L=1.5206657648086548,
    #             step_size=0.41058143973350525,
    #             # step_size=4.61,
    #             # L=4.670475,
    #             sqrt_diag_cov=1.0,
    #             initial_state=blackjax_state_after_tuning,
    #             return_ess_corr=True
    #         ),
    #         run_key,
    #         n=num_steps,
    #         batch=num_chains,
    #     )
    #     print(f"Effective Sample Size (ESS) of untuned adjusted mclmc with preconditioning set to {False} is avg {ess_avg} and max {ess} with acc rate of {acceptance_rate} and ess_corr_avg {ess_corr.mean().item()} and ess_corr_min {ess_corr.min().item()}")

    #     raise Exception

   
    ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
        model,
        run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune1=0.2, frac_tune2=0.2, frac_tune3=0.0, target_acc_rate=0.9, return_ess_corr=True, max=True, num_windows=1),
        run_key,
        n=num_steps,
        batch=num_chains,
    )
    print(f"Effective Sample Size (ESS) of tuned adjusted mclmc with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}, with L {params.L.mean()} and stepsize {params.step_size.mean()}")

    raise Exception
        
        
        # ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg = benchmark_chains(
        #     model,
        #     run_adjusted_mclmc(integrator_type=integrator_type, preconditioning=preconditioning, frac_tune1=0.1, frac_tune2=0.1, frac_tune3=0.5, target_acc_rate=0.9, return_ess_corr=True),
        #     run_key,
        #     n=num_steps,
        #     batch=num_chains,
        # )
        # print(f"Effective Sample Size (ESS) of tuned adjusted mclmc (stage 3) with preconditioning set to {preconditioning} is avg {ess_avg} and max {ess}, with L {params.L.mean()} and stepsize {params.step_size.mean()}")
        # print(f"acc rate is {acceptance_rate}")
        # print(f"ess corr is {ess_corr.min()}")
        # print(f"stage 2 L (max=False) is {params.L.mean()}")
        # print(f"optimal L avg is {np.sqrt(np.mean(model.E_x2)*model.ndims)}")
        # print(f"optimal L max is {np.sqrt(np.max(model.E_x2)*model.ndims)}")
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

        ess, ess_avg, ess_corr, params, acceptance_rate, grads_to_low_avg, _, _ = benchmark_chains(
            model,
            run_adjusted_mclmc_no_tuning(
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

    # optimal_L_for_gaussian(100, 10)

    # test_benchmarking()
    # raise Exception    

    models = {
    Gaussian(100, k, eigenvalues=eigenval_type): {'mclmc': 20000, 'adjusted_mclmc': 20000, 'nuts': 20000}
    for k in np.ceil(np.logspace(1, 5, num=10)).astype(int) for eigenval_type in ["log", "outliers"]
    }

    # Gaussian(d, condition_number=1., eigenvalues='linear'): {'mclmc': 20000, 'adjusted_mclmc': 20000, 'nuts': 20000}
    # for d in [2,3,4,5,6,7,8,9,10]
    # }

    # models = {

    #     Brownian(): {"mclmc": 40000, "adjusted_mclmc": 40000, "nuts": 40000},
    #     GermanCredit(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    #     ItemResponseTheory(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    #     Rosenbrock(): {'mclmc': 80000, 'adjusted_mclmc' : 80000, 'nuts': 80000},
    #     StochasticVolatility(): {'mclmc': 40000, 'adjusted_mclmc' : 40000, 'nuts': 40000},
    # }


    benchmark(batch_size=2, models=models, key_index=20, do_grid_search=False)
