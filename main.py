import jax
import jax.numpy as jnp
import gymnax
from memory_chain import MemoryChain, EnvParams

rng = jax.random.PRNGKey(9)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

env, env_params = MemoryChain(num_bits=2), EnvParams(max_steps_in_episode=1000, memory_length=5)
obs, state = env.reset(key_reset, env_params)
done = False

while not done:
    action = 0
    next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)
    print(obs, action, reward, next_obs)
    obs, state = next_obs, next_state
print(state)
print(obs, reward, done)