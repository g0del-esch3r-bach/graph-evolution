import random
import networkx as nx
import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt

N = 12
C = 0.9
MAX_EDGES = N * (N - 1) // 2
INI_POP = 200

CXPB = 0.7 
MUTPB = 0.8 

STALE_GEN = 10

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

def random_tree(num_nodes):

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

    leaf1 = leaves.pop(0)
    leaf2 = leaves.pop(0)
    edges.append((leaf1, leaf2))

    return edges

toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: random_tree(N))

def evaluate(individual):

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
        while not nx.is_connected(G):
            potential_edges = list(nx.non_edges(G))
            new_edge = random.choice(potential_edges)
            G.add_edge(*new_edge)
        return list(G.edges())

    ind1[:] = repair(child1_edges)
    ind2[:] = repair(child2_edges)
    return ind1, ind2

toolbox.register("mate", cxVariableEdges)

def mutToggleBestEdge(individual):

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

toolbox.register("select", tools.selTournament, tournsize=5)

def main():
    random.seed(42)

    pop = [toolbox.individual() for _ in range(INI_POP)]

    print("Starting Evolution Process:")

    stagnation_count = 0
    previous_best = None

    generation = 0
    while True:
        generation += 1

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values, child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        fits = [ind.fitness.values[0] for ind in pop]
        current_best = max(fits)

        print(f"Gen {generation}: Max Cost = {current_best:.4f}")

        if previous_best is not None and current_best == previous_best:
            stagnation_count += 1
        else:
            stagnation_count = 0
            previous_best = current_best

        if stagnation_count >= STALE_GEN:
            print(f"Terminating after {STALE_GEN} consecutive stagnant generations.")
            break

    best_individual = tools.selBest(pop, 1)[0]

    # Build the best graph
    G_best = nx.Graph()
    G_best.add_edges_from(best_individual)

    adj_matrix = nx.to_numpy_array(G_best, nodelist=range(N))

    alpha = C * (N + 1) / (N + 4)
    print(f"{N} {alpha}")
    for row in adj_matrix:
        print(" ".join(str(int(elem)) for elem in row))

    nx.draw(G_best, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.title("Optimized Graph (Highest Cost)")
    plt.show()

if __name__ == "__main__":
    main()