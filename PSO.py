import numpy as np
import warnings


# PSO Parameters
c_soc = 1.49445
c_cog = 1.49445
w = 0.729

def update_position(position, velocity, boundaries):
    new_position = position + velocity
    for i, b in enumerate(boundaries):
        if new_position[i] < b[0]:
            new_position[i] = b[0]
            velocity[i] = - np.random.random() * velocity[i]
        elif new_position[i] > b[1]:
            new_position[i] = b[1]
            velocity[i] = - np.random.random() * velocity[i]
    return new_position

# ----------------------------------

def update_velocity(position, velocity, global_best, local_best, max_velocities):
    n = len(velocity)
    r1 = np.random.random(n)
    r2 = np.random.random(n)
    social_component = c_soc * r1 * (global_best - position)
    cognitive_component = c_cog * r2 * (local_best - position)
    inertia = w*velocity
    new_velocity = inertia + social_component + cognitive_component
    for i, v in enumerate(max_velocities):
        if np.abs(new_velocity[i]) < v[0]:
            new_velocity[i] = np.sign(new_velocity[i]) * v[0]
        elif np.abs(new_velocity[i]) > v[1]:
            new_velocity[i] = np.sign(new_velocity[i]) * v[1]
    return new_velocity

# ----------------------------------

def pso_original(swarm_size, boundaries, alfa, n_iter, fit):
    
    max_velocities = [(-alfa * (up - down), alfa * (up - down)) for (down, up) in boundaries]
    positions = [np.array([np.random.random() * (b[1] - b[0]) + b[0] for b in boundaries])
                 for _ in range(0, swarm_size)]
    velocities = [np.array([np.random.choice([-1,1]) * np.random.uniform(v[0],v[1])
                            for v in max_velocities]) for _ in range(0, swarm_size)]
    local_best = positions
    global_best = min(positions, key=fit)
    hist = [positions]
    for _ in range(n_iter):
        velocities = [update_velocity(p, v, global_best, lb, max_velocities)
                      for p, v, lb in zip(positions, velocities, local_best)]
        positions = [update_position(p, v, boundaries) for p, v in zip(positions, velocities)]
        local_best = [min([p, lb], key=fit) for p, lb in zip(positions, local_best)]
        global_best = min([min(positions, key=fit), global_best], key=fit)
        hist.append(positions)
    return global_best, hist

# ----------------------------------

def update_velocity_evolved(position, velocity, global_best, local_best, max_velocities, update_rule):
    n = len(velocity)
    r1 = np.random.random(n)
    r2 = np.random.random(n)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            new_velocity = update_rule(position, velocity, global_best, local_best, r1, r2, w, c_cog, c_soc)

            # Penalization with zero velocity
            if np.isscalar(new_velocity) or new_velocity.shape != velocity.shape:
                return np.zeros_like(velocity)  
            
            if not np.all(np.isfinite(new_velocity)):
                return np.zeros_like(velocity)

    except Exception as e:
        print(f"Error in evolved rule: {e}")
        return np.zeros_like(velocity)  

    for i, v in enumerate(max_velocities):
        if np.abs(new_velocity[i]) < v[0]:
            new_velocity[i] = np.sign(new_velocity[i]) * v[0]
        elif np.abs(new_velocity[i]) > v[1]:
            new_velocity[i] = np.sign(new_velocity[i]) * v[1]
    return new_velocity

# -----------------------------------

def pso_evolved(swarm_size, boundaries, alfa, n_iter, fit, update_rule):

    max_velocities = [(-alfa * (up - down), alfa * (up - down)) for (down, up) in boundaries]
    positions = [np.array([np.random.uniform(b[0], b[1]) for b in boundaries]) for _ in range(swarm_size)]
    velocities = [np.array([np.random.uniform(v[0], v[1]) for v in max_velocities]) for _ in range(swarm_size)]
    local_best = positions[:]
    global_best = min(positions, key=fit)
    hist = [positions]

    for _ in range(n_iter):

        velocities = [update_velocity_evolved(p, v, global_best, lb, max_velocities, update_rule)
                      for p, v, lb in zip(positions, velocities, local_best)]
        positions = [update_position(p, v, boundaries) for p, v in zip(positions, velocities)]
        local_best = [min([p, lb], key=fit) for p, lb in zip(positions, local_best)]
        global_best = min([min(positions, key=fit), global_best], key=fit)
        hist.append(positions)

    return global_best, hist 