from .core import *
from .circstats import *

@jit
def components(arr: jnp.ndarray,
               p: float = 1.0,
               phi: jnp.ndarray = jnp.array([0.0]),
               axis: int | None,
               weights: jnp.ndarray = jnp.array([0])
               )
