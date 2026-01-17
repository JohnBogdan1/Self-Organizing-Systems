import random
import numpy as np

class Graph(object):
    def __init__(self, cost_matrix, rank):
        self.cost_matrix = cost_matrix
        self.rank = rank
        self.pheromone = [[1 / (rank ** 2) for i in range(rank)] for j in range(rank)]

class TSP_ACO(object):
    def __init__(self, ant_count, iterations, alpha, beta, rho, Q):
        self.ant_count = ant_count
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q

    def update_pheromone(self, graph, ants):
        for i in range(graph.rank):
            for j in range(graph.rank):
                graph.pheromone[i][j] *= self.rho
                for ant in ants:
                    graph.pheromone[i][j] += ant.pheromone_delta[i][j]

    def solve(self, graph):
        best_cost = float('inf')
        shortest_path = []

        for _ in range(self.iterations):
        	# generate ants
            ants = [Ant(self, graph, i) for i in range(self.ant_count)]
            for ant in ants:
                # calculate the total cost by visiting every node by this ant
                for _ in range(graph.rank - 1):
                    ant.select_next()

                # also. add to the cost the distance between last visited node and the initial one
                ant.total_cost += graph.cost_matrix[ant.tabu[-1]][ant.tabu[0]]

                # update best cost and best path
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    shortest_path = ant.tabu

                # update local pheromone for this ant
                ant.ant_update_pheromone()

            # update global pheromone after all ants finish the iteration
            self.update_pheromone(graph, ants)

        return shortest_path + [shortest_path[0]], best_cost

class Ant(object):
    def __init__(self, aco, graph, given_initial_node):
        self.ant_colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.tabu = []
        self.pheromone_delta = []
        self.allowed = [i for i in range(graph.rank)] # next component for each ant
        self.eta = [[0 if i == j else 1 / graph.cost_matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]
        initial_node = random.randint(0, graph.rank - 1)
        # initial_node = given_initial_node

        self.allowed.remove(initial_node)
        self.tabu.append(initial_node)
        self.current_node = initial_node

    def select_next(self):
        p_sum = 0
        for i in self.allowed:
            p_sum += self.graph.pheromone[self.current_node][i] ** self.ant_colony.alpha * self.eta[self.current_node][
                                                                                            i] ** self.ant_colony.beta
        probabilities = [0 for i in range(self.graph.rank)]
        for i in self.allowed:
            probabilities[i] = (self.graph.pheromone[self.current_node][i] ** self.ant_colony.alpha * \
                    self.eta[self.current_node][i] ** self.ant_colony.beta) / p_sum

        selected_node = None
        # 0 probability doesn't affect when i is not in 'allowed'
        # so, 0 probability node won't be chosen
        max_p = -9999
        for i, probability in enumerate(probabilities):
            if probability != 0 and probability > max_p:
                max_p = probability
                selected_node = i

        if selected_node in self.allowed:
            self.allowed.remove(selected_node)
        self.tabu.append(selected_node)
        self.total_cost += self.graph.cost_matrix[self.current_node][selected_node]
        self.current_node = selected_node

    def ant_update_pheromone(self):
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        for k in range(1, len(self.tabu)):
            i = self.tabu[k - 1]
            j = self.tabu[k]
            self.pheromone_delta[i][j] = self.ant_colony.Q / self.total_cost

def tsp_aco_run(graph, n, alpha, beta, rho):
    # print("The graph has %s nodes: %s" %(n, [i for i in range(n)]))
    # print(graph)

    aco_graph = Graph(graph, n)
    tsp_aco = TSP_ACO(ant_count=n, iterations=1000, alpha=alpha, beta=beta, rho=rho, Q=10.0)
    shortest_path, minimum_path_cost = tsp_aco.solve(aco_graph)

    return minimum_path_cost, shortest_path

if __name__ == '__main__':
    # n = 4
    # graph = generate_graph(n)

    # run tsp on graph
    # tsp_aco_run(graph, n)
    pass