import random
from functools import partial

import numpy as np
from copy import deepcopy
from random import sample, choice


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
    def __init__(self, p_id, initial_position, initial_velocity):
        self.p_id = p_id
        self.position = initial_position
        self.velocity = initial_velocity
        self.best_position = ()
        self.best_value = float('inf')

    def __str__(self):
        print("Particle %s. Current position" % self.p_id, self.position, "| My best_position is", self.best_position,
              "| My best_value is", self.best_value)


class ParticleSwarm(object):
    def __init__(self, number_of_particles, number_of_iterations, objective_function, boundaries, global_minimum, c,
                 inertia_weight=None, constriction_factor=False):
        self.number_of_particles = number_of_particles
        self.number_of_iterations = number_of_iterations
        self.boundaries = boundaries
        self.objective_function = objective_function
        self.global_minimum = global_minimum
        self.c = c
        self.inertia_weight = inertia_weight
        self.constriction_factor = constriction_factor
        self.particles = []
        self.swarm_best_position = []
        self.swarm_best_value = float('inf')
        self.max_velocity = np.array([10, 10])
        self.smp = 2
        self.spc = False
        self.cdc = 1
        self.srd = 0.1
        self.mr = 2

        # init and create particles
        interval = get_interval(boundaries[0], boundaries[1], self.number_of_particles)
        v_interval = get_interval(-self.max_velocity[0], self.max_velocity[1], self.number_of_particles)
        r = random.randint(0, self.number_of_particles - 1)

        self.swarm_best_position = np.array([interval[r], interval[r]])

        self.particles = [
            Particle(i, np.array([interval[i], interval[i]]),
                     np.array(v_interval[random.randint(0, self.number_of_particles - 1)],
                              v_interval[random.randint(0, self.number_of_particles - 1)])) for
            i in range(self.number_of_particles)]

    def optimize_function(self):
        iteration_counter = 0
        best_pos = None
        best_val = float('inf')

        j = None

        if self.spc:
            j = self.smp - 1
        else:
            j = self.smp

        mr_particles_population = self.mr_filtered_population(self.mr)

        while iteration_counter < self.number_of_iterations:
            self.update_swarm_best_particle_position()

            self.apply_behaviour(mr_particles_population, j)

            if best_val > self.swarm_best_value:
                best_pos = self.swarm_best_position
                best_val = self.swarm_best_value

            mr_particles_population = self.mr_filtered_population(self.mr)

            iteration_counter += 1

        return iteration_counter, best_pos, best_val

    def apply_behaviour(self, mr_particles_population, j):
        for i, particle in enumerate(self.particles):
            # in seeking mode
            if not mr_particles_population[i]:
                self.particles[i] = self.seek(particle, j)
            else:  # in tracing mode
                self.trace(particle)

    def seek(self, particle, j):
        candidates = []
        if self.spc:
            candidates.append(deepcopy(particle))

        particle_copies = [deepcopy(particle) for _ in range(j)]

        for particle_copy in particle_copies:
            selected_dims = sample(range(0, len(particle_copy.position)), int(self.cdc * len(particle_copy.position)))

            for dim in selected_dims:
                if np.random.uniform() < 0.5:
                    particle_copy.position[dim] += particle_copy.position[dim] * self.srd
                else:
                    particle_copy.position[dim] += particle_copy.position[dim] * self.srd * (-1)

            particle_copy.best_value = self.objective_function(particle_copy.position)
            candidates.append(particle_copy)

        are_all_equal = True
        for i in range(len(candidates) - 1):
            if candidates[i].best_value != candidates[i + 1].best_value:
                are_all_equal = False
                break

        chosen_candidate = None
        if are_all_equal:
            chosen_candidate = choice(candidates)
        else:
            FS = [candidate.best_value for candidate in candidates]
            FS_max = max(FS)
            FS_min = min(FS)
            FS_b = FS_max

            probabilities = np.array([abs(FS[i] - FS_b) / (FS_max - FS_min) for i in range(len(FS))])
            chosen_candidate = candidates[probabilities.argmax()]

        return chosen_candidate

    def trace(self, particle):
        self.update_velocity(particle)
        self.adjust_velocity(particle)
        self.update_position(particle)

    def update_swarm_best_particle_position(self):
        for particle in self.particles:
            best_evaluated_particle_position = self.objective_function(particle.position)
            if self.swarm_best_value > best_evaluated_particle_position:
                self.swarm_best_value = best_evaluated_particle_position
                self.swarm_best_position = particle.position

    def update_velocity(self, particle):
        c1 = self.c
        r1 = random.random()

        particle.velocity = particle.velocity + r1 * c1 * (self.swarm_best_position - particle.position)

    def adjust_velocity(self, particle):
        p_v = self.get_norm(particle.velocity)
        max_v = self.get_norm(self.max_velocity)

        if p_v < -max_v:
            particle.velocity = -self.max_velocity
        elif p_v > max_v:
            particle.velocity = self.max_velocity

    def update_position(self, particle):
        particle.position = particle.position + particle.velocity

    def mr_filtered_population(self, mr):
        n = len(self.particles)
        mr_pop = [False for _ in range(n)]
        temp_mr = deepcopy(mr)

        while True:
            r = random.randint(0, n - 1)

            if not mr_pop[r]:
                mr_pop[r] = True
                temp_mr -= 1

            if temp_mr == 0:
                break

        return mr_pop

    def get_norm(self, velocity):
        x = velocity[0]
        y = velocity[1]
        n = np.sqrt(x ** 2 + y ** 2)

        return n


if __name__ == '__main__':
    n = 20
    numbers_of_iterations = [50, 100, 500]
    functions = [(sphere_function, sphere_interval, sphere_minimum),
                 (rosenbrock_function, rosenbrock_interval, rosenbrock_minimum),
                 (rastrigin_function, rastrigin_interval, rastrigin_minimum),
                 (griewank_function, griewank_interval, griewank_minimum)]

    run_once = False

    c = 0.5

    if run_once:
        best_pos_solution = None
        best_value_solution = float('inf')
        best_iteration_counter_solution = None
        best_values = []
        average_best_val = None

        for _ in range(10):
            pso = ParticleSwarm(n, numbers_of_iterations[2], rosenbrock_function, rosenbrock_interval,
                                rosenbrock_minimum,
                                c, None, False)

            iteration_counter, best_pos, best_val = pso.optimize_function()

            best_values.append(best_val)

            if best_value_solution > best_val:
                best_pos_solution = best_pos
                best_value_solution = best_val
                best_iteration_counter_solution = iteration_counter

        average_best_val = sum(best_values) / len(best_values)
        print("The best position is:", best_pos_solution, "| The best value is:", best_value_solution,
              "| The average best value is:", average_best_val,
              "| In %s iterations" % best_iteration_counter_solution)

    else:
        for my_function in functions:
            print("*", my_function[0].__name__)
            for number_of_iterations in numbers_of_iterations:
                print("**", number_of_iterations)

                best_pos_solution = None
                best_value_solution = float('inf')
                best_iteration_counter_solution = None
                best_values = []
                average_best_val = None

                for _ in range(30):
                    pso = ParticleSwarm(n, number_of_iterations, my_function[0], my_function[1], my_function[2],
                                        c, None, False)

                    iteration_counter, best_pos, best_val = pso.optimize_function()

                    best_values.append(best_val)

                    if best_value_solution > best_val:
                        best_pos_solution = best_pos
                        best_value_solution = best_val
                        best_iteration_counter_solution = iteration_counter

                average_best_val = sum(best_values) / len(best_values)
                print("The best position is:", best_pos_solution, "| The best value is:", best_value_solution,
                      "| The average best value is:", average_best_val,
                      "| In %s iterations" % best_iteration_counter_solution)
            print()
