import numpy as np
from jax import jit
import jax.numpy as jnp

def _pluralize(noun, count):
    return noun if count == 1 else f"{noun}s"

def _raisedim_check(*input_values:np.ndarray | jnp.ndarray):

    messages = [
        f"{input_value.ndim} {_pluralize('dimension', input_value.ndim)}"
        for input_value in input_values
    ]
    if len(messages) == 1:
        message = messages[0]
    elif len(messages) == 2:
        message = f"{messages[0]} and {messages[1]}"
    else:
        message = "those inputs"
    raise ValueError(f"Not sure what to do with {message}")
