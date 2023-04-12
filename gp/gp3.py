import random
import numpy as np
import operator
from deap import creator, base, tools, gp, algorithms
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Generate the dataset for linear regression
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

# 2. Define the fitness function
def evaluate_linear_regression(individual, X, y):
    func = toolbox.compile(expr=individual)
    y_pred = func(X[:, 0]) # func(X) returns a 2D array, so we need to take the first column
    mse = mean_squared_error(y, y_pred)
    return mse,

# 3. Create the types, individuals, and the genetic programming pipeline
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray], np.ndarray)
pset.addPrimitive(np.add, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(np.subtract, [np.ndarray, np.ndarray], np.ndarray)
pset.addPrimitive(np.multiply, [np.ndarray, np.ndarray], np.ndarray)
pset.addTerminal(1, np.ndarray)
pset.renameArguments(ARG0="X")

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_linear_regression, X=X, y=y)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# 4. Run the genetic programming pipeline
def main():
    random.seed(42)
    pop = toolbox.population(n=300)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats, halloffame=hof, verbose=True)
    return hof

if __name__ == "__main__":
    hof = main()
    print("Best individual:", hof[0])
    best_func = toolbox.compile(expr=hof[0])
    y_pred = best_func(X)
    print("Predicted values:", y_pred)
