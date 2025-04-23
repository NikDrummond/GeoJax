### basis.py
# basis.py
# Cartesian basis vector definitions

import jax.numpy as jnp

class Basis:
    """
    Standard 3D Cartesian basis vectors.
    """
    @property
    def x(self) -> jnp.ndarray:
        return jnp.array([1.0, 0.0, 0.0])

    @property
    def y(self) -> jnp.ndarray:
        return jnp.array([0.0, 1.0, 0.0])

    @property
    def z(self) -> jnp.ndarray:
        return jnp.array([0.0, 0.0, 1.0])

    @property
    def neg_x(self) -> jnp.ndarray:
        return jnp.array([-1.0, 0.0, 0.0])

    @property
    def neg_y(self) -> jnp.ndarray:
        return jnp.array([0.0, -1.0, 0.0])

    @property
    def neg_z(self) -> jnp.ndarray:
        return jnp.array([0.0, 0.0, -1.0])


basis = Basis()
