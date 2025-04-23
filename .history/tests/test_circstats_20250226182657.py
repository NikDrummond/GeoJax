import jax.numpy as jnp
import jax.random as jr
import jax
import numpy as np
from astropy import _components, _angle, circmean, _length, circvar, circstd
import pytest
from GeoJax import vector_geometry

data = jnp.array(np.linspace(-np.pi, np.pi))
weights = jnp.array(np.random.uniform(0,1,len(data)))

### circmean tests

def test_mean_radians_unweighted()
    data = jnp.array(np.)


