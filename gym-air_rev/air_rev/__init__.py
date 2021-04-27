from gym.envs.registration import register

register(
    id='air_rev-v0',
    entry_point='air_rev.envs:AirRev',
    max_episode_steps=100000
)