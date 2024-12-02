import dm_env
from dm_env import specs
import numpy as np
from camera import Camera

class XArmEnvironment(dm_env.Environment):

    def __init__(self):
        self._camera = Camera(fps=60)
    
    def reset(self) -> dm_env.TimeStep:
        return dm_env.restart(self._observation())
    
    def step(self, action) -> dm_env.TimeStep:
        """Updates the environment according to the action."""
        return dm_env.transition(reward=0.0, observation=self._observation())

    # TODO implement specs
    def action_spec(self):
        """Returns the action spec."""
        return specs.DiscreteArray(
            dtype=int, num_values=1, name="action"
        )
    
    def observation_spec(self):
        """Returns the observation spec."""
        return specs.BoundedArray(
            shape=(1,),
            dtype=np.float32,
            name="camera",
            minimum=0.0,
            maximum=1.0
        )

    def _observation(self) -> np.ndarray:
        # return self._camera._get_frame()
        return 0.0
