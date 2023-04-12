import random
import numpy as np
from deap import base, creator, tools, algorithms, gp

sample_size = 100
max_depth = 5
population_size = 100
generations = 100
crossover_prob = 0.5
mutation_prob = 0.2
tournament_size = 3

# generate data
X = np.random.uniform(-1, 1, sample_size)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, sample_size)

# define fitness function
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# define individual
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

## define toolbox
toolbox = base.Toolbox()
# define primitive set
pset = gp.PrimitiveSet("MAIN", 1) # defines the number of arguments of the expression 
toolbox.register("attr_bool", random.randint, 0, 1) # defines the type of individual
#toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2) # defines the type of individual
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool) # defines the type of individual
toolbox.register("population", tools.initRepeat, list, toolbox.individual) # defines the type of population

# define genetic operators
#toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=tournament_size)

# define evaluation function
def eval(individual, X, y):
    func = toolbox.compile(expr=individual)
    y_pred = np.array([func(x) for x in X])
    return np.mean((y - y_pred)**2),

def eval_regression(individual, X, y):
    y_pred = np.dot(X, individual)
    return np.mean((y - y_pred)**2),

# register evaluation function
toolbox.register("evaluate", eval, X=X, y=y)

# define main function
def main():
    pop = toolbox.population(n=population_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_prob, mutpb=mutation_prob, ngen=generations, stats=stats, halloffame=hof, verbose=True)

    return pop, log, hof

if __name__ == "__main__":
    pop, log, hof = main()
    print(hof[0])
    print(log)