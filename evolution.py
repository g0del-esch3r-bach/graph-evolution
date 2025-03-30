import random
import networkx as nx
import numpy as np
from deap import base, creator, tools

N_NODES = 10
MAX_EDGES = N_NODES * (N_NODES - 1) // 2  # Maximum number of edges in an undirected graph
NGEN = 100  # <-- Moved here
INI_POP = 2000  # <-- New variable for initial population size

# Step 1: Fitness and Individual Definitions
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Generate random connected graph with variable edges
def random_connected_graph(n_nodes):
    """Generate a random connected graph represented as a list of edges."""
    while True:
        num_edges = random.randint(n_nodes - 1, MAX_EDGES)  # At least N-1 edges to ensure potential connectivity
        G = nx.gnm_random_graph(n_nodes, num_edges)
        if nx.is_connected(G):
            return list(G.edges())

toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: random_connected_graph(N_NODES))

# NOTE: We could still register a population function if desired:
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# But in this example, we'll manually generate the population inside main() using INI_POP.

# Fitness Evaluation: Maximize average shortest-path length
def evaluate(individual):
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    G.add_edges_from(individual)

    if nx.is_connected(G):
        avg_dist = nx.average_shortest_path_length(G)
        return (avg_dist,)
    else:
        # Penalize disconnected graphs significantly
        return (-1000,)

toolbox.register("evaluate", evaluate)

# Crossover Operator: Edge-based crossover allowing variable edge counts
def cxVariableEdges(ind1, ind2):
    edge_set1, edge_set2 = set(ind1), set(ind2)
    common = edge_set1 & edge_set2
    diff1, diff2 = list(edge_set1 - common), list(edge_set2 - common)

    # Shuffle and split
    random.shuffle(diff1)
    random.shuffle(diff2)
    split1 = len(diff1) // 2
    split2 = len(diff2) // 2

    child1_edges = list(common) + diff1[:split1] + diff2[split2:]
    child2_edges = list(common) + diff2[:split2] + diff1[split1:]

    def repair(edges):
        G = nx.Graph()
        G.add_nodes_from(range(N_NODES))
        G.add_edges_from(edges)
        # If disconnected, add edges randomly until connected
        while not nx.is_connected(G):
            potential_edges = set(nx.non_edges(G))
            new_edge = random.choice(list(potential_edges))
            G.add_edge(*new_edge)
        return list(G.edges())

    ind1[:], ind2[:] = repair(child1_edges), repair(child2_edges)
    return ind1, ind2

# Mutation Operator: Randomly add or remove an edge, ensuring connectivity
def mutVariableEdge(individual):
    G = nx.Graph()
    G.add_nodes_from(range(N_NODES))
    G.add_edges_from(individual)

    mutation_type = random.choice(['add', 'remove'])

    if mutation_type == 'add':
        potential_edges = list(nx.non_edges(G))
        if potential_edges:
            new_edge = random.choice(potential_edges)
            G.add_edge(*new_edge)
    elif mutation_type == 'remove':
        if len(G.edges) > N_NODES - 1:  # Ensure at least N-1 edges remain
            edge_to_remove = random.choice(list(G.edges))
            G.remove_edge(*edge_to_remove)
            # If removal causes disconnection, revert
            if not nx.is_connected(G):
                G.add_edge(*edge_to_remove)

    return (list(G.edges()),)

toolbox.register("mate", cxVariableEdges)
toolbox.register("mutate", mutVariableEdge)
toolbox.register("select", tools.selTournament, tournsize=3)

# Run GA Optimization
def main():
    random.seed(42)

    # Create initial population of size INI_POP
    # Using the random_connected_graph function for each individual.
    pop = [toolbox.individual() for _ in range(INI_POP)]

    CXPB, MUTPB = 0.7, 0.2

    print("Starting Evolution Process:")
    for gen in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        # Mutation
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate fitness of invalid individuals
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Statistics per Generation
        fits = [ind.fitness.values[0] for ind in pop]
        mean_fit = sum(fits) / len(pop)
        max_fit = max(fits)
        print(f"Gen {gen}: Max Avg Distance = {max_fit:.2f}, Avg = {mean_fit:.2f}")

    # Final Results
    best_individual = tools.selBest(pop, 1)[0]
    print("\nBest Graph Edges:", best_individual)
    print("Best Average Shortest Path:", best_individual.fitness.values[0])

    # Visualize Best Graph
    import matplotlib.pyplot as plt
    G_best = nx.Graph()
    G_best.add_edges_from(best_individual)
    nx.draw(G_best, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Optimized Graph with Variable Edges")
    plt.show()

if __name__ == "__main__":
    main()
