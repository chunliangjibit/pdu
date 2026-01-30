# verify_p1_setup.py
import jax
import jax.numpy as jnp
from pdu.core.types import GasState, ParticleState, State
from pdu.thermo.implicit_eos import get_thermo_state

def test_types():
    gas = GasState(rho=jnp.array([1.8]), u=jnp.array([0.0]), T=jnp.array([300.0]), lam=jnp.array([0.0]))
    part = ParticleState(phi=jnp.array([[0.2]]), rho=jnp.array([[2.7]]), u=jnp.array([[0.0]]), T=jnp.array([[300.0]]), r=jnp.array([[10e-6]]))
    state = State(x=jnp.array([0.0]), gas=gas, part=part)
    print("Types verification: SUCCESS")
    return state

def test_implicit_eos():
    rho = jnp.array(1.8)
    e = jnp.array(1e6)
    params = {}
    res = get_thermo_state(rho, e, params)
    print(f"Implicit EOS verification (Result): {res}")
    
    # Test gradient
    def simple_loss(r, e):
        res = get_thermo_state(r, e, params)
        return jnp.sum(res.P)
    
    grad_fn = jax.grad(simple_loss, argnums=(0, 1))
    grads = grad_fn(rho, e)
    print(f"Implicit EOS verification (Grads): {grads}")

if __name__ == "__main__":
    test_types()
    test_implicit_eos()
