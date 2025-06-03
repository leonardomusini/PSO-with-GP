from deap import base, creator, gp, tools
import numpy as np
import random
import operator
import multiprocessing
from functools import partial

from PSO import pso_evolved
from functions import training_functions
from utils import valid_add, valid_sub, valid_mul, valid_div


# GP Parameters
POP_SIZE = 100         # Population size
N_GEN = 50             # Number of generations
CX_PROB = 0.9          # Crossover probability
MUT_PROB = 0.5         # Mutation probability
TOURNAMENT_SIZE = 3    # Tournament size for selection
N_ELITES = 1           # Number best individuals to keep in the next generation

# DEAP Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

# GP Primitive Set
pset = gp.PrimitiveSet("MAIN", 9)

# Basic arithmetic functions
pset.addPrimitive(valid_add, 2)
pset.addPrimitive(valid_sub, 2)
pset.addPrimitive(valid_mul, 2)
pset.addPrimitive(valid_div, 2)

# PSO variables
pset.renameArguments(ARG0='position')
pset.renameArguments(ARG1='velocity')
pset.renameArguments(ARG2='global_best')
pset.renameArguments(ARG3='local_best')
pset.renameArguments(ARG4='r1')
pset.renameArguments(ARG5='r2')
pset.renameArguments(ARG6='w')
pset.renameArguments(ARG7='c_cog')
pset.renameArguments(ARG8='c_soc')

# Additional terminals for PSO
pset.addTerminal(-1.0)
pset.addTerminal(-0.5)
pset.addTerminal(1.0)
pset.addTerminal(0.5)
pset.addEphemeralConstant("rand", partial(np.random.uniform, -1.0, 1.0))

# Toolbox for GP evolution
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=10)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", pso_evolved)
toolbox.register("mate", gp.cxOnePoint)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr, pset=pset)
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
toolbox.register("select", tools.selTournament, tournsize=TOURNAMENT_SIZE)

# -----------------------------------

def evaluate_gp(individual, mode=1, n_iter=10, problem_instances=training_functions, pars_press=0.01):

    evolved_rule = toolbox.compile(expr=individual)

    # Parsimony pressure on the size of the tree
    tree_size = len(individual)
    penalty = pars_press * tree_size

    total_fitness = 0

    for problem in problem_instances:

        problem_fitness_sum = 0
        
        for i in range(n_iter):

            np.random.seed(i)
            best, hist = pso_evolved(
                swarm_size=30,
                boundaries=problem["boundaries"],
                alfa=0.2,
                n_iter=50,
                fit=lambda p: problem["fit"](*p),
                update_rule=evolved_rule
            )

            if mode == 1:
                # Fitness as difference of objective function values between best and global minimum
                fitness_value = np.abs(problem["fit"](*best) - problem["fit"](*problem["global_min"]))
            elif mode == 2:
                # Fitness as distance of best position from global minimum
                fitness_value = np.sum(np.abs(best - problem["global_min"]))
            elif mode == 3:
                # Fitness as sum of distances of all particles from global minimum
                final_positions = hist[-1]
                fitness_value = np.sum(np.sum(np.abs(p - problem["global_min"])) for p in final_positions)

            problem_fitness_sum += fitness_value

        # Average fitness for this problem
        avg_problem_fitness = problem_fitness_sum / n_iter
        total_fitness += avg_problem_fitness

    # Average fitness across all problems
    avg_fitness = total_fitness / len(problem_instances)
    avg_fitness += penalty  

    return avg_fitness

# -----------------------------------

def train_gp(n_gen=N_GEN, pop_size=POP_SIZE, cx_prob=CX_PROB, mut_prob=MUT_PROB, n_elites=N_ELITES, mode=1, multiproc=True):

    min_fitness = []
    avg_fitness = []
    max_fitness = []

    if multiproc:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)
    else:
        toolbox.register("map", map)

    # Creation of the intitial population
    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)  
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    # Evaluation of the initial population
    fitnesses = toolbox.map(partial(evaluate_gp, mode=mode), population)
    for ind, fit_val in zip(population, fitnesses):
        ind.fitness.values = (fit_val,)

    # Evolutionary Loop
    for gen in range(n_gen):

        # SELECTION
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        # CROSSOVER
        #random.shuffle(offspring)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # MUTATION
        for mutant in offspring:
            if np.random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # FITNESS EVALUATION
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(partial(evaluate_gp, mode=mode), invalid_ind)
        for ind, fit_val in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit_val,)

        # ELITISM
        elite_individuals = tools.selBest(population, n_elites)
        offspring = tools.selWorst(offspring, len(offspring) - n_elites) + elite_individuals

        population[:] = offspring
        hof.update(population)

        record = stats.compile(population)
        min_fitness.append(record['min'])
        avg_fitness.append(record['avg'])
        max_fitness.append(record['max'])
        print(f"Gen {gen} - Min: {record['min']}, Avg: {record['avg']}, Max: {record['max']}")

    if multiproc:
        pool.close()
        pool.join()

    print("Best evolved rule:", hof[0])
    return hof[0], min_fitness, avg_fitness