from utils import generate_graph, show_graph_with_labels
from tsp import tsp_run
from aco_tsp import tsp_aco_run

alpha_arr = [2.0, 1.0, 5.0, 4.0, 3.0, 2.5, 7.0, 8.0]
beta_arr = [1.5, 2.0, 3.0, 1.0, 5.0, 1.5, 4.0, 1.0]
rho_arr = [0.25, 0.2, 0.5, 0.3, 0.5, 0.3, 0.6, 0.8]
run_multiple = True

if __name__ == '__main__':
    n = 9

    graph = generate_graph(n)
    show_graph_with_labels(graph, n)

	# run naive tsp
    print("----------NAIVE TSP----------")
    minimum_path_cost, shortest_path = tsp_run(graph, n)
    print("The minimum route cost is:", minimum_path_cost)
    print("The shortest path is:", shortest_path)

    print("\n#########################################################\n")

    if run_multiple:
        for i in range(8):
            alpha = alpha_arr[i]
            beta = beta_arr[i]
            rho = rho_arr[i]
            costs = []
            for j in range(10):
                # run aco tsp
                print("Iteration (%s, %s)" % (i, j))
                print("----------ACO TSP----------")
                minimum_path_cost, shortest_path = tsp_aco_run(graph, n, alpha, beta, rho)
                costs.append(minimum_path_cost)
                #print("The minimum route cost is:", minimum_path_cost)
                #print("The shortest path is:", shortest_path)
            print("The minimum route cost is:", min(costs))
    else:
        print("----------ACO TSP----------")
        minimum_path_cost, shortest_path = tsp_aco_run(graph, n, alpha_arr[0], beta_arr[0], rho_arr[0])
        print("The minimum route cost is:", minimum_path_cost)
        print("The shortest path is:", shortest_path)