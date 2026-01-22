from utils import generate_graph
import random

initial_node = 0

def naive_tsp(graph, visited, current_node, n, visited_count, cost, paths, current_path):
    if visited_count == n and graph[current_node][initial_node] != 0:
    	final_cost = cost + graph[current_node][initial_node]

    	# if there are more paths with the same cost
    	# doesnn't matter, because we are interested in the shortest one
    	# if there are many shortest, just pick the last assigned one
    	paths[final_cost] = current_path + [initial_node]
    	return

    # iterate through the neighbors like this, because is complete graph
    for neighbor_node in range(n):
        # if neighbor of node is not not visited
        if not visited[neighbor_node] and current_node != neighbor_node:
            # mark neighbor as visited
            visited[neighbor_node] = True

            naive_tsp(graph, visited, neighbor_node, 
            	n, visited_count + 1, cost + graph[current_node][neighbor_node], 
            	paths, current_path + [neighbor_node])

            # after, unmark it
            # so other nodes different than current_node can visit it further
            visited[neighbor_node] = False

def tsp_run(graph, n):
    print("The graph has %s nodes: %s" %(n, [i for i in range(n)]))
    print(graph)

    paths = {}
    visited = [False for i in range(n)]
    visited[initial_node] = True

    # initial_node = random.randint(0, n - 1)
    print("Starting node:", initial_node)

    naive_tsp(graph, visited, initial_node, n, 1, 0, paths, [initial_node])

    minimum_path_cost = min(paths.keys())

    # print("The minimum route cost is:", minimum_path_cost)
    shortest_path = paths[minimum_path_cost]
    # print("The shortest path is:", shortest_path)

    return minimum_path_cost, shortest_path


if __name__ == '__main__':
    # n = 4
    # graph = generate_graph(n)

    # run tsp on graph
    # tsp_run(graph, n)
    pass
