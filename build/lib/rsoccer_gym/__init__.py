from gym.envs.registration import register

register(id='VSS-v1',
         entry_point='rsoccer_gym.vss.env_vss:VSSEnv',
         max_episode_steps=10000
         )
