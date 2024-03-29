import random
import numpy as np
import argparse
from dataclasses import dataclass

"""
Reference: 
Michael Chun-Yu Niu, Michael Niu, "Composite Airframe Structures", Hong Kong Conmilit Press Ltd. (2005)
Summary of Stacking Sequence Design Considerations:
1. Use balanced and symmetric layups
2. Intersperse ply orientations
"If possible, avoid grouping 90° plies; separate them by 0° or ±45° 
  (0° is direction of critical load) to  minimize interlaminar shear and normal stresses.
  and whenever possible maintain a homogeneous stacking sequence and avoid 
  grouping or similar plies. If plies must be grouped, avoid grouping more 
  than 5 plies of the same oriantation together to minimaze edge splitting."
3. Minimize groupings of same orientation
4. Alternate +45° and -45° plies through the layup
5. Separate groups of same orientation by 45° plies
6. Provide at least 10% of each of the four ply orientation (10% design rule)
7. Careful designs needed at locations prone to delamination (e.g. avoiding tape plies with fibers perpendicular to an edge)
8. Exterior surface plies should be continuous and ±45° to the primary load direction (e.g. locating 0° plies at least 3 plies from the surface).
"Design to reduce Poisson's ratio: 
    - consider the use of 90° plies in a laminate
    - reduce the % of 0° plies
    - reduce of Poisson's ration is critical in bonded parts"
"""
@dataclass(frozen=True)
class DesignParameters:
    orientations = [0, 45, -45, 90] # standard orientations
    min_percentage: float = .1 # minimum 10% of plies of each orientation
    max_group_size: int = 3 # maximum number of consecutive plies with the same orientation
    max_imbalance: int = 2 # maximum difference between the number of plies with 0° and 90°

@dataclass(frozen=False)
class GAdefault:
    n_plies: int = 16 #default number of plies
    population_size: int = 150 # default population size
    maximum_iterations: int = 1000 # default maximum number of iterations
    mutation_rate: float = .1 # default mutation rate
    cros_rate: float = .7 # defautl crossover rate
    penalty_factor: float = 2 #  penalty weight to use in fitness function


parser= argparse.ArgumentParser()

# Number of plies in each stacking sequence
parser.add_argument('--num_plies', type=int, required=False, default=GAdefault.n_plies, help="int: number of plies in the stacking sequence")

# Mutation and Crossover rate 
parser.add_argument('--mut_rate', type=np.float64, required=False, default=GAdefault.mutation_rate, help="float: mutation rate")
parser.add_argument('--cross_rate', type=np.float64, required=False, default=GAdefault.cros_rate, help="float: crossover rate")

# Population size, max number of iterations
parser.add_argument('--pop_size', type=int, required=False, default=GAdefault.population_size, help="int: maximum number of stackings generated")
parser.add_argument('--num_iter', type=int, required=False, default=GAdefault.maximum_iterations, help="int: maximum number of iterations")

# Parse the arguments
args=parser.parse_args()
NUM_PLIES = int(args.num_plies)
MUTATION_RATE = np.float64(args.mut_rate)
CROSSOVER_RATE = np.float64(args.cross_rate)
POP_SIZE = int(args.pop_size)
NUM_ITERATIONS = int(args.num_iter)

# fitness function
def fitness_function(sequence):
    """
    Fitness Function. Each sequence is evaluated against the set of rules and a fitness score is return.
    Arguments:
    sequence: list, stacking sequence of the laminate.
    Returns:
    fitness: float, fitness score of the sequence.
    """
    # Rule 1: Symmetry check respect to mid-plane
    symmetry_penalty = 0
    for i in range(len(sequence)//2):
        if sequence[i] != sequence[-i-1]:
            symmetry_penalty += 1

    # Rule 2: Calculate the percentage of each orientation
    counts = {orientation: sequence.count(orientation) for orientation in DesignParameters.orientations}
    for orientation in DesignParameters.orientations:
        if counts[orientation] / len(sequence) < DesignParameters.min_percentage:
            return 0
    percentages = {orientation: counts[orientation] / len(sequence) for orientation in DesignParameters.orientations}
    
    # Rule 3: Calculate the number of groupings of plies with the same orientation
    groups = 0
    last_orientation = None
    group_size = 0
    for orientation in sequence:
        if orientation == last_orientation:
            group_size += 1
        else:
            groups += max(0, group_size - 1)
            last_orientation = orientation
            group_size = 1
    groups += max(0, group_size - 1)
    
    # Rule 4: Calculate the imbalance between 0° and 90°
    imbalance = abs(counts[0] - counts[90])
    
    # Rule 5: Penalize plies with fibers perpendicular to a free edge at the mid-plane
    edge_penalty = 0
    for i in range(len(sequence)):
        if sequence[i] in {0, 90}:
            if i == 0 or i == len(sequence) - 1:
                edge_penalty += 1
            elif sequence[i-1] != sequence[i] and sequence[i+1] != sequence[i]:
                edge_penalty += 1
    
    # Rule 6: Penalty for not alternating +45° and -45° plies
    alternating_penalty = 0
    last_angle = None
    for orientation in sequence:
        if orientation in {45, -45}:
            if last_angle is None:
                last_angle = orientation
            elif last_angle == -orientation:
                last_angle = orientation
            else:
                alternating_penalty += 1
        elif orientation == 0 or orientation == 90:
            last_angle = None
            
    # Rule 7: Penalty for grouping tape plies with the same orientation without a 45° ply in between
    grouping_penalty = 0
    last_orientation = None
    last_group_size = 0
    for orientation in sequence:
        if orientation == last_orientation:
            last_group_size += 1
        else:
            if last_group_size > 1 and last_orientation is not None:
                if sequence.index(orientation) - last_group_size < 0:
                    before = 45
                else:
                    before = sequence[sequence.index(orientation) - last_group_size - 1]
                if sequence.index(orientation) + 1 >= len(sequence):
                    after = 45
                else:
                    after = sequence[sequence.index(orientation) + 1]
                if abs(before - last_orientation) != 45 and abs(after - last_orientation) != 45:
                    grouping_penalty += 1
            last_orientation = orientation
            last_group_size = 1
    if last_group_size > 1 and last_orientation is not None:
        if abs(sequence[sequence.index(last_orientation) - last_group_size] - last_orientation) != 45:
            grouping_penalty += 1
            
    # Rule 8: Penalty for 0° plies too close to the surface
    surface_penalty = 0
    for i in range(len(sequence)):
        if sequence[i] == 0:
            if i < 3 or i > len(sequence) - 4:
                surface_penalty += 1
            elif 0 in sequence[i-3:i] or 0 in sequence[i+1:i+4]:
                surface_penalty += 1
    
    # Calculate the fitness value
    fitness = 1 / (1 + symmetry_penalty*GAdefault.penalty_factor + percentages[0] + percentages[45] + percentages[-45] + percentages[90] + groups +
                   imbalance + edge_penalty + grouping_penalty + surface_penalty)

    return fitness

# Mutation Function:
def mutate(sequence, mut_rate, n_plies):
    mutated_seq = sequence.copy()
    
    # Check if the sequence is already at its maximum length
    if len(mutated_seq) == n_plies:
        # Rule 1: Swap adjacent plies
        i = np.random.randint(len(sequence)-1)
        mutated_seq[i], mutated_seq[i+1] = mutated_seq[i+1], mutated_seq[i]

        # Rule 2: Randomly choose an orientation from the available orientations
        i = np.random.randint(len(sequence))
        mutated_seq[i] = np.random.choice(DesignParameters.orientations)

        # Rule 5: Swap ply at the mid-plane with ply 45° off-axis
        mid_plane = len(sequence) // 2
        if abs(mutated_seq[mid_plane]) == 90:
            if np.random.random() < mut_rate and 45 in sequence:
                for i in range(mid_plane+1, len(sequence)):
                    if mutated_seq[i] == 45:
                        mutated_seq[mid_plane], mutated_seq[i] = mutated_seq[i], mutated_seq[mid_plane]
                        break
            else:
                for i in reversed(range(mid_plane)):
                    if abs(mutated_seq[i]) == 45:
                        mutated_seq[mid_plane], mutated_seq[i] = mutated_seq[i], mutated_seq[mid_plane]
                        break

        # Rule 6: Swap closest plies to mid-plane that are both θ° or -θ°
        if np.random.random() < mut_rate:
            for i in range(mid_plane-1):
                if abs(mutated_seq[i]) == abs(mutated_seq[i+1]):
                    mutated_seq[i], mutated_seq[i+1] = mutated_seq[i+1], mutated_seq[i]
                    break
            for i in reversed(range(mid_plane+1, len(sequence)-1)):
                if abs(mutated_seq[i]) == abs(mutated_seq[i+1]):
                    mutated_seq[i], mutated_seq[i+1] = mutated_seq[i+1], mutated_seq[i]
                    break

        # Rule 7: Separate groups of tape plies of the same orientation with 45° plies
        for i in range(1, len(sequence)-1):
            if mutated_seq[i] == mutated_seq[i-1] and mutated_seq[i] == mutated_seq[i+1]:
                if abs(mutated_seq[i-1]) + abs(mutated_seq[i+1]) == 90:
                    if mutated_seq[i] == 0:
                        mutated_seq[i], mutated_seq[i+1] = mutated_seq[i+1], mutated_seq[i]
                    else:
                        mutated_seq[i], mutated_seq[i-1] = mutated_seq[i-1], mutated_seq[i]

        # Rule 8: Move 0° plies at the surface closer to the middle of the sequence
        if 0 in mutated_seq:
            surface = mutated_seq.index(0)
            if surface < 3:
                for i in range(surface+1, len(sequence)):
                    if mutated_seq[i] == 0 and i - surface >= 3:
                        # Swap the current 0° ply with the ply 3 positions closer to the middle
                        middle = len(mutated_seq) // 2
                        swap_index = max(surface+2, middle)
                        mutated_seq[swap_index], mutated_seq[i] = mutated_seq[i], mutated_seq[swap_index]
                        break
                    elif mutated_seq[i] != 0:
                        break
    return mutated_seq

# Selection function (tournament selection)
def selection(fitness_values, population_size):
    parents_indices = []
    for _ in range(population_size):
        indices = random.sample(range(len(fitness_values)), 2)
        if fitness_values[indices[0]] > fitness_values[indices[1]]:
            parents_indices.append(indices[0])
        else:
            parents_indices.append(indices[1])
    return parents_indices

# Crossover function (single-point crossover)
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        index = random.randint(1, len(parent1) - 1)
        child = parent1[:index] + parent2[index:]
    else:
        child = parent1
    return child

# Initial population
def generate_population(num_plies, pop_size):
    """
    This function generates the initial population of stacking sequences. To ensure 
    a population with symmetryc layups, symmetry constraints is enforced by flipping the orientation of the i-th ply
     if the (num_plies - i - 1)-th ply has the same orientation
    """
    population = []
    for i in range(pop_size):
        individual = [random.choice(DesignParameters.orientations) for _ in range(num_plies)]
        for j in range(num_plies // 2):
            if individual[j] == individual[num_plies - j - 1]:
                individual[num_plies - j - 1] = random.choice(DesignParameters.orientations)
        population.append(individual)
    return population

# Genetic Algorithm
def genetic_sequence(number_of_plies, population_size, max_iterations, crossover_rate, mutation_rate):
    # Initial population
    population = generate_population(number_of_plies, population_size)

    for i in range(max_iterations):
        # fitness value of each sequence
        fitnesses = [fitness_function(sequence) for sequence in population]
        # select parents for crossover function
        parent_indices = selection(fitnesses,population_size)

        # generate new population
        new_population = []
        for j in range(population_size // 2):
            parent1 = population[parent_indices[2*j]]
            parent2 = population[parent_indices[2*j+1]]
            if np.random.random() < crossover_rate:
                child1 = crossover(parent1, parent2, crossover_rate)
                child2 = crossover(parent2, parent1, crossover_rate)
            else:
                child1 = parent1
                child2 = parent2
            new_population.append(child1)
            new_population.append(child2)
        
        # Apply mutations to the new population
        for j in range(population_size):
            if np.random.random() < mutation_rate:
                new_population[j] = mutate(new_population[j], mutation_rate,number_of_plies)
            # Replace the old population with the new population
        population = new_population
    
    return max(population, key=fitness_function)

if __name__ == "__main__":
    best_sequence= genetic_sequence(NUM_PLIES, POP_SIZE, NUM_ITERATIONS, CROSSOVER_RATE, MUTATION_RATE)
    print(f'Best fitness = {fitness_function(best_sequence)}, Best sequence = {best_sequence}, Sequence Length: {len(best_sequence)}')


