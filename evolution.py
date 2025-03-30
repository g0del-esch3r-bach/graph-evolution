import random
import networkx as nx
import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
N = 20
C = 0.9
MAX_EDGES = N * (N - 1) // 2
INI_POP = 200

CXPB = 0.7  # Probability of crossover
MUTPB = 0.8 # Probability of mutation

# We no longer have NGEN
# Instead, we define a 'STALE_GEN'
STALE_GEN = 10  # e.g., if the max cost doesn't improve for 5 gens, stop.

# -----------------------------------------------------------------------------
# DEAP Setup
# -----------------------------------------------------------------------------
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# -----------------------------------------------------------------------------
# 1) Generate a Random Tree via Prufer-code
# -----------------------------------------------------------------------------
def random_tree(num_nodes):
    """
    Generate a random tree on [0..num_nodes-1] using a random Prufer code.
    Return a list of edges (u, v).
    """
    if num_nodes <= 1:
        return []

    length = num_nodes - 2
    prufer = [random.randrange(num_nodes) for _ in range(length)]

    node_count = [0] * num_nodes
    for node in prufer:
        node_count[node] += 1

    leaves = [i for i in range(num_nodes) if node_count[i] == 0]
    leaves.sort()

    edges = []
    for node in prufer:
        leaf = leaves.pop(0)
        edges.append((leaf, node))
        node_count[node] -= 1
        if node_count[node] == 0:
            import bisect
            bisect.insort(leaves, node)

    # Connect the final two leaves
    leaf1 = leaves.pop(0)
    leaf2 = leaves.pop(0)
    edges.append((leaf1, leaf2))

    return edges

toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: random_tree(N))

# -----------------------------------------------------------------------------
# 2) Fitness Evaluation — maximize a negative cost
# -----------------------------------------------------------------------------
def evaluate(individual):
    """
    Build a graph from 'individual' (list of edges).
    If connected, compute cost = -(3*alpha*avgDist/(N+1)) - (2*(1-alpha)*E/(N*(N-1))).
    If disconnected, return -1000.
    """
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(individual)

    if nx.is_connected(G):
        alpha = C * (N + 1) / (N + 4)
        avglen = nx.average_shortest_path_length(G)
        edges = G.number_of_edges()
        cost = -(3 * alpha * avglen / (N + 1)) - (2 * (1 - alpha) * edges / (N * (N - 1)))
        return (cost,)
    else:
        return (-1000,)

toolbox.register("evaluate", evaluate)

# -----------------------------------------------------------------------------
# 3) Crossover Operator
# -----------------------------------------------------------------------------
def cxVariableEdges(ind1, ind2):
    edge_set1, edge_set2 = set(ind1), set(ind2)
    common = edge_set1 & edge_set2
    diff1 = list(edge_set1 - common)
    diff2 = list(edge_set2 - common)

    random.shuffle(diff1)
    random.shuffle(diff2)
    split1 = len(diff1) // 2
    split2 = len(diff2) // 2

    child1_edges = list(common) + diff1[:split1] + diff2[split2:]
    child2_edges = list(common) + diff2[:split2] + diff1[split1:]

    def repair(edges):
        G = nx.Graph()
        G.add_nodes_from(range(N))
        G.add_edges_from(edges)
        # If disconnected, add edges until connected
        while not nx.is_connected(G):
            potential_edges = list(nx.non_edges(G))
            new_edge = random.choice(potential_edges)
            G.add_edge(*new_edge)
        return list(G.edges())

    ind1[:] = repair(child1_edges)
    ind2[:] = repair(child2_edges)
    return ind1, ind2

toolbox.register("mate", cxVariableEdges)

# -----------------------------------------------------------------------------
# 4) Mutation Operator — Toggle the single best edge
# -----------------------------------------------------------------------------
def mutToggleBestEdge(individual):
    """
    For each possible edge in the complete graph on N nodes, toggle it
    in a copy of the individual's edges. Evaluate the cost. 
    Pick the toggle that yields the highest (least negative) cost.
    If that best toggle is > old cost, adopt it; otherwise, do nothing.
    Exactly one toggle or none is performed.
    """
    def get_cost(edges_list):
        return toolbox.evaluate(edges_list)[0]

    old_edges = set(individual)
    old_cost = get_cost(individual)

    best_cost = old_cost
    best_toggle_edge = None
    toggle_is_adding = False

    all_possible_edges = [(i, j) for i in range(N) for j in range(i+1, N)]

    for e in all_possible_edges:
        if e in old_edges:
            new_edges = old_edges - {e}
        else:
            new_edges = old_edges | {e}

        new_edges_list = list(new_edges)
        new_cost = get_cost(new_edges_list)

        if new_cost > best_cost:
            best_cost = new_cost
            best_toggle_edge = e
            toggle_is_adding = (e not in old_edges)

    if best_cost > old_cost:
        if toggle_is_adding:
            individual[:] = list(old_edges | {best_toggle_edge})
        else:
            individual[:] = list(old_edges - {best_toggle_edge})

    return (individual,)

toolbox.register("mutate", mutToggleBestEdge)

# -----------------------------------------------------------------------------
# 5) Selection Operator
# -----------------------------------------------------------------------------
toolbox.register("select", tools.selTournament, tournsize=5)

# -----------------------------------------------------------------------------
# 6) GA Optimization — Terminate After X Consecutive Gens with Same Max
# -----------------------------------------------------------------------------
def main():
    random.seed(42)

    # Create initial population of size INI_POP
    pop = [toolbox.individual() for _ in range(INI_POP)]

    print("Starting Evolution Process:")

    # For tracking stagnation
    stagnation_count = 0
    previous_best = None

    generation = 0
    while True:
        generation += 1

        # Select parents
        offspring = toolbox.select(pop, len(pop))
        # Clone them
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

        # Evaluate fitness of any invalid offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace old population with offspring
        pop[:] = offspring

        # Compute stats
        fits = [ind.fitness.values[0] for ind in pop]
        current_best = max(fits)

        print(f"Gen {generation}: Max Cost = {current_best:.4f}")

        # Check if best cost is the same as the previous generation's best
        if previous_best is not None and current_best == previous_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            previous_best = current_best

        # If we have stagnated for STALE_GEN consecutive gens, stop
        if stagnation_count >= STALE_GEN:
            print(f"Terminating after {STALE_GEN} consecutive stagnant generations.")
            break

    # Final Results
    best_individual = tools.selBest(pop, 1)[0]

    # Build the best graph
    G_best = nx.Graph()
    G_best.add_edges_from(best_individual)

    # Convert best graph to adjacency matrix
    adj_matrix = nx.to_numpy_array(G_best, nodelist=range(N))

    alpha = C * (N + 1) / (N + 4)
    print(f"{N} {alpha}")
    for row in adj_matrix:
        print(" ".join(str(int(elem)) for elem in row))

    # Also visualize
    nx.draw(G_best, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Optimized Graph (Highest Cost)")
    plt.show()

if __name__ == "__main__":
    main()
