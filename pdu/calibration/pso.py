
import jax
import jax.numpy as jnp
import numpy as np
from typing import Callable, Tuple, Optional

class PSOOptimizer:
    """
    [V10 Algorithm Layer] JAX-accelerated Particle Swarm Optimization.
    Designed for global search in high-dimensional thermodynamic parameter spaces.
    """
    def __init__(
        self, 
        objective_fn: Callable, 
        lower_bounds: jnp.ndarray, 
        upper_bounds: jnp.ndarray,
        num_particles: int = 50,
        c1: float = 1.5, # Cognitive coefficient
        c2: float = 1.5, # Social coefficient
        w: float = 0.5,  # Inertia weight
    ):
        self.objective_fn = jax.jit(objective_fn)
        self.lb = lower_bounds
        self.ub = upper_bounds
        self.num_particles = num_particles
        self.dim = lower_bounds.shape[0]
        self.c1 = c1
        self.c2 = c2
        self.w = w

    def optimize(self, num_iterations: int = 100, seed: int = 42):
        key = jax.random.PRNGKey(seed)
        k1, k2, k3 = jax.random.split(key, 3)
        
        # 1. Initialize Particles
        pos = jax.random.uniform(k1, (self.num_particles, self.dim), minval=self.lb, maxval=self.ub)
        # Scale velocity by search range
        range_vec = self.ub - self.lb
        vel = jax.random.uniform(k2, (self.num_particles, self.dim), minval=-0.2*range_vec, maxval=0.2*range_vec)
        
        # 2. Initial Evaluation
        fitness = jax.vmap(self.objective_fn)(pos)
        fitness = jnp.nan_to_num(fitness, nan=1e12, posinf=1e12, neginf=1e12)
        
        pbest_pos = pos
        pbest_fit = fitness
        
        gbest_idx = jnp.argmin(fitness)
        gbest_pos = pos[gbest_idx]
        gbest_fit = fitness[gbest_idx]
        
        # Hyperparameters (Optimized for convergence)
        w_pso = 0.8
        c1_pso = 2.0
        c2_pso = 2.0

        def iter_body(i, state):
            (pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, key) = state
            
            k1, k2, key = jax.random.split(key, 3)
            r1 = jax.random.uniform(k1, (self.num_particles, self.dim))
            r2 = jax.random.uniform(k2, (self.num_particles, self.dim))
            
            # Update Velocity
            new_vel = w_pso * vel + \
                      c1_pso * r1 * (pbest_pos - pos) + \
                      c2_pso * r2 * (gbest_pos - pos)
            
            # Update Position
            new_pos = pos + new_vel
            # Clamping to bounds
            new_pos = jnp.clip(new_pos, self.lb, self.ub)
            
            # Evaluate Fitness
            new_fitness = jax.vmap(self.objective_fn)(new_pos)
            new_fitness = jnp.nan_to_num(new_fitness, nan=1e10, posinf=1e10, neginf=1e10)
            
            # Update PBest
            improved = new_fitness < pbest_fit
            # Reshape to (N, 1) for broadcasting
            improved_exp = jnp.expand_dims(improved, 1)
            
            pbest_pos = jnp.where(improved_exp, new_pos, pbest_pos)
            pbest_fit = jnp.where(improved, new_fitness, pbest_fit)
            
            # Update GBest
            best_idx = jnp.argmin(pbest_fit)
            curr_gbest_fit = pbest_fit[best_idx]
            
            gbest_pos = jnp.where(curr_gbest_fit < gbest_fit, pbest_pos[best_idx], gbest_pos)
            gbest_fit = jnp.minimum(curr_gbest_fit, gbest_fit)
            
            return (new_pos, new_vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, key)

        state = (pos, vel, pbest_pos, pbest_fit, gbest_pos, gbest_fit, k3)
        final_state = jax.lax.fori_loop(0, num_iterations, iter_body, state)
        
        return final_state[4], final_state[5] # gbest_pos, gbest_fit

def fit_jwl_pso_bridge(V_data, P_data, weight_isentrope=1.0):
    """
    Bridge function to use PSO for JWL fitting.
    Optimizes (A, B, R1, R2, w, C).
    """
    
    # Objective: Log-MSE between JWL and data
    def jwl_objective(params):
        A, B, R1, R2, w, C = params
        
        # Physical constraints penalty (Hard constraint via infinity fitness)
        # R1 > R2 > w
        # R1 > 1.0, R2 > 0.1
        penalty = jnp.where(R1 <= R2 + 0.1, 1e10, 0.0)
        penalty += jnp.where(R2 <= w, 1e10, 0.0)
        penalty += jnp.where(w > 1.5, 1e10, 0.0)
        
        P_pred = A * jnp.exp(-R1 * V_data) + \
                 B * jnp.exp(-R2 * V_data) + \
                 C / (V_data**(1.0 + w))
        
        mse = jnp.mean((jnp.log(P_pred) - jnp.log(P_data))**2)
        return mse + penalty

    # Bounds: A[0, 5000], B[0, 200], R1[1, 12], R2[0.1, 5], w[0.05, 1.5], C[0.01, 20]
    lb = jnp.array([10.0, 0.1, 1.5, 0.2, 0.05, 0.1])
    ub = jnp.array([5000.0, 300.0, 15.0, 5.0, 1.5, 50.0])
    
    optimizer = PSOOptimizer(jwl_objective, lb, ub, num_particles=100)
    best_params, best_fit = optimizer.optimize(num_iterations=200)
    
    return best_params, best_fit

if __name__ == "__main__":
    # Smoke test
    print("PSO Optimizer Smoke Test...")
    def sphere_fn(x): return jnp.sum(x**2)
    lb = jnp.array([-5.0, -5.0])
    ub = jnp.array([5.0, 5.0])
    pso = PSOOptimizer(sphere_fn, lb, ub)
    best_x, best_y = pso.optimize(50)
    print(f"Sphere Best X: {best_x}, Best Y: {best_y}")
    assert best_y < 1e-3
    print("PSO Smoke Test Passed!")
