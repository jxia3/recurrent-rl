import jax
import jax.numpy as jnp
import gymnax

rng = jax.random.PRNGKey(0)
rng, key_reset, key_act, key_step = jax.random.split(rng, 4)

env, env_params = gymnax.make("MemoryChain-bsuite")
obs, state = env.reset(key_reset, env_params)
done = False

while not done:
    action = 1
    next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)
    print(obs, action, reward, next_obs)
    obs, state = next_obs, next_state
print(state)
print(obs, reward, done)