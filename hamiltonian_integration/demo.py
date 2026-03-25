import jax
import jax.numpy as jnp
import blackjax
import numpy as np


import jax.scipy.stats as stats
from datetime import date


rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

loc, scale = 10,10

observed = np.random.normal(loc, scale, size=1_000)

def log_density_fn(loc, log_scale, observed=observed):
    # Univariate normal
    scale = jnp.exp(log_scale)
    logjac = log_scale
    logpdf = stats.norm.logpdf(observed, loc, scale)
    return logjac + jnp.sum(logpdf)


logdensity = lambda x: log_density_fn(**x)

## HMC ## 

inv_mass_matrix = np.array([0.5,0.01])
num_integration_steps = 60
step_size = 1e-3

hmc = blackjax.hmc(logdensity, step_size, inv_mass_matrix, num_integration_steps)

initial_position = {"loc": 1.0, "log_scale": 1.0}

initial_state = hmc.init(initial_position)



def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states

# Build the kernel

hmc_kernel = jax.jit(hmc.step)

rng_key, sample_key = jax.random.split(rng_key)
states = inference_loop(sample_key, hmc_kernel, initial_state, 10_000)

mcmc_samples = states.position
mcmc_samples["scale"] = jnp.exp(mcmc_samples["log_scale"]).block_until_ready()