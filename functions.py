import numpy as np


# TRAINING FUNCTIONS

def sphere(x, y):
    return x**2 + y**2

def rosenbrock(x, y):
    return 100 * (y - x**2)**2 + (1 - x)**2

def rastrigin(x, y):
    return 20 + (x**2 - 10 * np.cos(2 * np.pi * x)) + (y**2 - 10 * np.cos(2 * np.pi * y))

def ackley(x, y):
    return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x**2 + y**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

training_functions = [
    {"name": "Sphere", "fit": sphere, "boundaries": [(-100, 100), (-100, 100)], "global_min": np.array((0, 0))},
    {"name": "Rosenbrock", "fit": rosenbrock, "boundaries": [(-5, 10), (-5, 10)], "global_min": np.array((1, 1))},
    {"name": "Rastrigin", "fit": rastrigin, "boundaries": [(-5.12, 5.12), (-5.12, 5.12)], "global_min": np.array((0, 0))},
    {"name": "Ackley", "fit": ackley, "boundaries": [(-5, 5), (-5, 5)], "global_min": np.array((0, 0))}   
]

# TESTING FUNCTIONS

def michalewicz(x, y):
    m = 10
    return -np.sin(x) * np.sin(x**2 / np.pi)**(2 * m) - np.sin(y) * np.sin(2 * y**2 / np.pi)**(2 * m)

def griewank(x, y):
    return 1 + (x**2 / 4000) + (y**2 / 4000) - np.cos(x) * np.cos(y / np.sqrt(2))

def levy(x, y):
    w1 = 1 + (x - 1) / 4
    w2 = 1 + (y - 1) / 4
    return np.sin(np.pi * w1)**2 + ((w1 - 1)**2) * (1 + 10 * np.sin(np.pi * w1 + 1)**2) + ((w2 - 1)**2) * (1 + np.sin(2 * np.pi * w2)**2)

def schwefel(x, y):
    return 418.9829 * 2 - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))

def zakharov(x, y):
    term1 = x**2 + y**2
    term2 = (0.5 * x + 0.5 * y)**2
    term3 = (0.5 * x + 0.5 * y)**4
    return term1 + term2 + term3

testing_functions = [
    {"name": "Michalewicz", "fit": michalewicz, "boundaries": [(0, np.pi), (0, np.pi)], "global_min": np.array((2.20, 1.57))},
    {"name": "Griewank", "fit": griewank, "boundaries": [(-600, 600), (-600, 600)], "global_min": np.array((0, 0))},
    {"name": "Levy", "fit": levy, "boundaries": [(-10, 10), (-10, 10)], "global_min": np.array((1, 1))},
    {"name": "Schwefel", "fit": schwefel, "boundaries": [(-500, 500), (-500, 500)], "global_min": np.array((420.9687, 420.9687))},
    {"name": "Zakharov", "fit": zakharov, "boundaries": [(-5, 10), (-5, 10)], "global_min": np.array((0, 0))}
]