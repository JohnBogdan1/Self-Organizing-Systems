import random
import numpy as np


def generate_full_graph(n):
    graph = {i: [[j for j in range(n) if i != j], [None, None]] for i in range(n)}

    return graph


def generate_ring_graph(n):
    graph = {i: [[n - 1 if i == 0 else (i - 1), (i + 1) if i < (n - 1) else 0], [None, None]] for i in range(n)}

    return graph


def generate_square_graph(n):
    graph = {i: [[(i + j + 1) % n for j in range(4)], [None, None]] for i in range(n)}

    return graph


def get_interval(start, end, n):
    return np.linspace(start, end, n)


sphere_minimum = 0
sphere_interval = [-5.12, 5.12]


def sphere_function(position):
    s = 0
    d = len(position)

    for i in range(d):
        s += position[i] ** 2

    return s


rosenbrock_minimum = 0
rosenbrock_interval = [5, 10]


def rosenbrock_function(position):
    s = 0
    d = len(position)

    for i in range(d - 1):
        s += 100 * (position[i + 1] - position[i] ** 2) ** 2 + (position[i] - 1) ** 2

    return s


rastrigin_minimum = 0
rastrigin_interval = [-5.12, 5.12]


def rastrigin_function(position):
    s = 0
    d = len(position)

    for i in range(d):
        s += position[i] ** 2 - 10 * np.cos(2 * np.pi * position[i])

    return s + 10 * d


griewank_minimum = 0
griewank_interval = [-600, 600]


def griewank_function(position):
    s = 0
    p = 1
    d = len(position)

    for i in range(d):
        s += position[i] ** 2 / 4000
        p *= np.cos(position[i] / np.sqrt(i + 1))

    return s - p + 1


class Particle(object):
    def __init__(self, p_id, initial_position):
        self.p_id = p_id
        self.position = initial_position
        self.velocity = np.array([0, 0])
        self.best_position = ()
        self.best_value = float('inf')

    def __str__(self):
        print("Particle %s. Current position" % self.p_id, self.position, "| My best_position is", self.best_position,
              "| My best_value is", self.best_value)


class ParticleSwarm(object):
    def __init__(self, number_of_particles, number_of_iterations, topology, objective_function, boundaries,
                 global_minimum, pair, inertia_weight=None, constriction_factor=False):
        self.number_of_particles = number_of_particles
        self.number_of_iterations = number_of_iterations
        self.topology = topology
        self.boundaries = boundaries
        self.objective_function = objective_function
        self.global_minimum = global_minimum
        self.pair = pair
        self.inertia_weight = inertia_weight
        self.constriction_factor = constriction_factor
        self.particles = []
        self.swarm_best_position = []
        self.swarm_best_value = float('inf')

        # init and create particles
        interval = get_interval(boundaries[0], boundaries[1], self.number_of_particles)
        r = random.randint(0, self.number_of_particles - 1)

        self.swarm_best_position = np.array([interval[r], interval[r]])

        # set it in topology, too
        for node in self.topology:
            self.topology[node][1] = [self.swarm_best_position, self.swarm_best_value]

        self.particles = [Particle(i, np.array([interval[i], interval[i]])) for i in range(self.number_of_particles)]

    def optimize_function(self):
        iteration_counter = 0
        epsilon = 1e-5

        best_pos = None
        best_val = None
        while iteration_counter < self.number_of_iterations:
            self.update_particle_best_position()
            self.update_swarm_best_particle_position()

            for node in self.topology:
                if abs(self.topology[node][1][1] - self.global_minimum) < epsilon:
                    best_pos = self.topology[node][1][0]
                    best_val = self.topology[node][1][1]
                    return iteration_counter, best_pos, best_val

            self.update_velocities()
            self.update_positions()

            iteration_counter += 1

        # return the best we found, after all iterations
        best_pos = None
        best_val = float('inf')
        for node in self.topology:
            if best_val > self.topology[node][1][1]:
                best_pos = self.topology[node][1][0]
                best_val = self.topology[node][1][1]

        return iteration_counter, best_pos, best_val

    def update_particle_best_position(self):
        for particle in self.particles:
            evaluated_particle_position = self.objective_function(particle.position)
            if particle.best_value > evaluated_particle_position:
                particle.best_value = evaluated_particle_position
                particle.best_position = particle.position

    def update_swarm_best_particle_position(self):
        for particle in self.particles:
            best_evaluated_particle_position = self.objective_function(particle.position)
            if self.topology[particle.p_id][1][1] > best_evaluated_particle_position:
                self.topology[particle.p_id][1][1] = best_evaluated_particle_position
                self.topology[particle.p_id][1][0] = particle.position

                for neighbor in self.topology[particle.p_id][0]:
                    self.topology[neighbor][1][1] = best_evaluated_particle_position
                    self.topology[neighbor][1][0] = particle.position

    def update_velocities(self):
        alpha_1 = pair[0]
        alpha_2 = pair[1]

        if self.inertia_weight == None:
            w = 1.0
        else:
            w = self.inertia_weight

        if self.constriction_factor == False:
            X = 1.0
        else:
            X = (alpha_1 + alpha_2) / 2

        for particle in self.particles:
            u_1 = random.random()
            u_2 = random.random()

            if self.constriction_factor == False:
                particle.velocity = w * particle.velocity + \
                                    alpha_1 * u_1 * (particle.best_position - particle.position) + \
                                    alpha_2 * u_2 * (self.topology[particle.p_id][1][0] - particle.position)
            else:
                particle.velocity = X * (particle.velocity +
                                         alpha_1 * u_1 * (particle.best_position - particle.position) +
                                         alpha_2 * u_2 * (self.topology[particle.p_id][1][0] - particle.position))

    def update_positions(self):
        for particle in self.particles:
            particle.position = particle.position + particle.velocity

    def print_particles(self):
        for particle in self.particles:
            particle.__str__()


if __name__ == '__main__':
    n = 20
    number_of_iterations = 500

    topologies = [generate_full_graph, generate_ring_graph, generate_square_graph]
    variants = [[None, False], [0.5, False], [None, True]]
    pairs = [[0.1, 0.3], [0.5, 0.7], [0.6, 0.4]]

    variant_map = ['Main algorithm', 'Intertia Weight', 'Constriction Factor']

    run_once = True

    if run_once:
        topology = generate_square_graph(n)

        best_pos_solution = None
        best_value_solution = float('inf')
        best_iteration_counter_solution = None

        pair = pairs[1]

        for _ in range(1):

            pso = ParticleSwarm(n, number_of_iterations, topology, rosenbrock_function, rosenbrock_interval,
                                rosenbrock_minimum, pair, None, False)

            # pso.print_particles()

            iteration_counter, best_pos, best_val = pso.optimize_function()

            # print("The best position is:", best_pos, "| The best value is:", best_val, "| In %s iterations" % iteration_counter)

            if best_value_solution > best_val:
                best_pos_solution = best_pos
                best_value_solution = best_val
                best_iteration_counter_solution = iteration_counter

        print("The best position is:", best_pos_solution, "| The best value is:", best_value_solution,
              "| In %s iterations" % best_iteration_counter_solution)

    else:
        for topology_function in topologies:
            topology = topology_function(n)

            print("*", topology_function.__name__)

            for variant in variants:
                print("**", variant_map[variants.index(variant)])
                for pair in pairs:
                    print("***", pair)
                    best_pos_solution = None
                    best_value_solution = float('inf')
                    best_iteration_counter_solution = None

                    for _ in range(10):

                        pso = ParticleSwarm(n, number_of_iterations, topology, sphere_function, sphere_interval,
                                            sphere_minimum, pair, variant[0], variant[1])

                        # pso.print_particles()

                        iteration_counter, best_pos, best_val = pso.optimize_function()

                        # print("The best position is:", best_pos, "| The best value is:", best_val, "| In %s iterations" % iteration_counter)

                        if best_value_solution > best_val:
                            best_pos_solution = best_pos
                            best_value_solution = best_val
                            best_iteration_counter_solution = iteration_counter

                    print("The best position is:", best_pos_solution, "| The best value is:", best_value_solution,
                          "| In %s iterations" % best_iteration_counter_solution)
                print()
            print()
