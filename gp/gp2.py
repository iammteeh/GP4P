import random
import numpy as np
from deap import base, creator, tools, algorithms

# Generate sample data for a simple linear regression problem
n_samples = 100
X = np.linspace(0, 1, n_samples).reshape(-1, 1) # Reshape X to be a 2D array
y = 2*X + np.random.normal(scale=0.1, size=n_samples)

# Define genetic algorithm parameters
POPULATION_SIZE = 50
N_GENERATIONS = 10
CXPB = 0.5  # crossover probability
MUTPB = 0.1  # mutation probability

# Define fitness function
def eval_regression(individual, X, y):
    individual = np.array(individual).reshape(-1, 1)
    y_pred = np.dot(X, individual)
    return np.mean((y - y_pred)**2),

# Create DEAP toolbox
toolbox = base.Toolbox()

# Define individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define genetic operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run genetic algorithm
pop = toolbox.population(n=POPULATION_SIZE)
for gen in range(N_GENERATIONS):
    offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
    fits = toolbox.map(lambda x: eval_regression(x, X, y), offspring)
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    pop = toolbox.select(offspring, k=len(pop))
best_ind = tools.selBest(pop, k=1)[0]
print("Best individual:", best_ind)


# then do symbolic regression to get a regression tree
#REGRESSION = "symbolic"
#X_train = X_train[new_features]
#X_test = X_test[new_features]
#with Regression(X_train, X_test, y_train, REGRESSION) as regression:
#    regression.fit()
#    program = regression.get_program()
#    model.program = program
#    print(f"regression program: {program}")
#    print(model.plot_symbolic_program())
