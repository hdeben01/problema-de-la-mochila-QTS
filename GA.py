
import random
from pathlib import Path

def generate_random_value():
    return random.randint(0, 1)

def create_individual(n_items):
    return [generate_random_value() for _ in range(n_items)]

def compute_fitness(chromosome, values, weights, max_weight):
    value = sum([chromosome[i] * values[i] for i in range(len(chromosome))])
    weight = sum([chromosome[i] * weights[i] for i in range(len(chromosome))])
    if weight > max_weight:
        return 0
    else:
        return value

def compute_weight(chromosome, weights):
    return sum([chromosome[i] * weights[i] for i in range(len(chromosome))])

def selection(population, fitness_scores):
    selected_chromosomes = []
    population_size = len(population)
    # Si todos los fitness son cero (población totalmente inviable), seleccion aleatoria
    if max(fitness_scores) == 0:
        return random.sample(population, population_size // 2)
    for i in range(population_size // 2):
        max_fitness_index = fitness_scores.index(max(fitness_scores))
        selected_chromosomes.append(population[max_fitness_index])
        fitness_scores[max_fitness_index] = 0
    return selected_chromosomes

def crossover(parent1, parent2):
    split_index = random.randint(1, len(parent1)-1)
    child1 = parent1[:split_index] + parent2[split_index:]
    child2 = parent2[:split_index] + parent1[split_index:]
    return child1, child2

def mutate(chromosome, mutation_rate):
    for i in range(len(chromosome)):
        if random.uniform(0, 1) < mutation_rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

def create_feasible_individual(n_items, values, weights, max_weight):
    """
    Genera un individuo factible usando un enfoque goloso aleatorio.
    """
    individual = [0]*n_items
    idxs = list(range(n_items))
    random.shuffle(idxs)
    current_weight = 0
    for i in idxs:
        if current_weight + weights[i] <= max_weight:
            individual[i] = 1
            current_weight += weights[i]
    return individual

def genetic_algorithm(file_path, population_size=100, generations=100, mutation_rate=0.1):
    """
    Ejecuta el algoritmo genético del problema de la mochila leyendo la instancia desde un archivo.

    Args:
        file_path (str): ruta al archivo de datos de la instancia.
        population_size (int): tamaño de la población.
        generations (int): número de generaciones.
        mutation_rate (float): probabilidad de mutación.

    Returns:
        List[int]: historial de fitness máximo por generación.
    """
    n_items, values, weights, max_weight = load_input_from_file(file_path)
    # create the initial population
    population = []
    for _ in range(population_size):
        ind = create_individual(n_items)
        # si no es factible, sustituir por individuo goloso aleatorio
        if compute_fitness(ind, values, weights, max_weight) == 0:
            ind = create_feasible_individual(n_items, values, weights, max_weight)
        population.append(ind)
    historial_soluciones = []

    # run the genetic algorithm for the specified number of generations
    for generation in range(generations):
        # calculate the fitness of each chromosome in the population
        fitness_scores = [compute_fitness(chromosome, values, weights, max_weight) for chromosome in population]
        #print(fitness_scores)
        max_fitness_index = fitness_scores.index(max(fitness_scores))
        historial_soluciones.append(fitness_scores[max_fitness_index])
        # select the top chromosomes for reproduction
        selected_chromosomes = selection(population, fitness_scores)

        # crossover the selected chromosomes to create new offspring
        offspring = []
        for i in range(population_size // 2):
            parent1 = selected_chromosomes[random.randint(0, len(selected_chromosomes)-1)]
            parent2 = selected_chromosomes[random.randint(0, len(selected_chromosomes)-1)]
            child1, child2 = crossover(parent1, parent2)
            offspring.extend([child1, child2])

        # mutate the offspring
        for i in range(len(offspring)):
            offspring[i] = mutate(offspring[i], mutation_rate)

        # replace the old population with the new offspring
        population = offspring

    # find the chromosome with the highest fitness score
    best_chromosome = population[0]
    best_fitness_score = compute_fitness(best_chromosome, values, weights, max_weight)
    for chromosome in population:
        fitness_score = compute_fitness(chromosome, values, weights, max_weight)
        if fitness_score > best_fitness_score:
            best_chromosome = chromosome
            best_fitness_score = fitness_score

    # return the solution
    selected_items = [i+1 for i in range(n_items) if best_chromosome[i] == 1]
    print(best_chromosome)
    solution = {
        'items': selected_items,
        'value': best_fitness_score,
        'weight': compute_weight(best_chromosome, weights)
    }
    return solution, historial_soluciones


def load_input_from_file(file_path):
    """
    Lee datos de un archivo en disco con el formato:
      n <n_items>
      c <max_weight>
      z 0
      time 0.00
      idx,value,weight,0
      ...

    Args:
        file_path (str): ruta al archivo de datos.

    Returns:
        Tuple[int, List[int], List[int], int]: n_items, values, weights, max_weight
    """
    values = []
    weights = []
    n_items = 0
    max_weight = 0

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Cabecera
    idx = 0
    while idx < len(lines) and ' ' in lines[idx]:
        key, val = lines[idx].split()
        if key == 'n':
            n_items = int(val)
        elif key == 'c':
            max_weight = int(val)
        idx += 1
    # Ítems
    for line in lines[idx:]:
        parts = line.split(',')
        if len(parts) >= 3:
            values.append(int(parts[1]))
            weights.append(int(parts[2]))
    return n_items, values, weights, max_weight

instancia_mochila = Path('.\data\knapPI_11_500_1000_1.csv')

solution,historial_soluciones = genetic_algorithm(instancia_mochila,1000,1000,0.5)

print(solution)