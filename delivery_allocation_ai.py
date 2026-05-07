import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

def create_environment():
    """
    Creates a 10x10 grid environment.
    0 represents free space, 1 represents obstacles.
    """
    grid = np.zeros((10, 10))
    grid[2:8, 2] = 1
    grid[2:8, 7] = 1
    grid[5, 3:7] = 1
    return grid

def bfs_shortest_path(grid, start, goal):
    """
    Breadth-First Search (BFS) to find the shortest path cost from start to goal.
    """
    rows, cols = grid.shape

    queue = deque([(start, 0)])
    visited = set()
    visited.add(start)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current, dist = queue.popleft()

        if current == goal:
            return dist

        for dr, dc in directions:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if grid[nr, nc] == 0 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append(((nr, nc), dist + 1))

    return float('inf')

def init_population(pop_size, chromo_length):
    """Generates an initial population of random binary arrays."""
    return [[random.choice([0, 1]) for _ in range(chromo_length)] for _ in range(pop_size)]

def calculate_fitness(chromosome, distances):
    """
    Calculates the fitness of a chromosome.
    Aim: minimize the difference between Vehicle 1 and Vehicle 2 distances.
    """
    dist_v1 = 0
    dist_v2 = 0

    for i in range(len(chromosome)):
        if chromosome[i] == 0:
            dist_v1 += distances[i]
        else:
            dist_v2 += distances[i]

    difference = abs(dist_v1 - dist_v2)
    fitness_score = 1.0 / (1.0 + difference)

    return fitness_score, dist_v1, dist_v2

def tournament_selection(population, fitnesses):
    """Selects the best parent out of a random sample of 3."""
    selected_indices = random.sample(range(len(population)), 3)
    best_index = max(selected_indices, key=lambda idx: fitnesses[idx])
    return population[best_index]

def single_point_crossover(parent1, parent2):
    """
    Splits parents at a random point and swaps their tails.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(chromosome, mutation_rate, gene_min=0, gene_max=1):
    """Random mutation: reset a gene to a random value within bounds."""
    for i in range(len(chromosome)):
        if random.random() < mutation_rate:
            chromosome[i] = random.randint(gene_min, gene_max)
    return chromosome

def genetic_algorithm(
    distances,
    pop_size=20,
    generations=50,
    mutation_rate=0.1,
    patience=8,
    target_fitness=1.0
):
    """
    Main GA loop to find the best package allocation.
    """
    chromo_length = len(distances)
    population = init_population(pop_size, chromo_length)

    best_solution = None
    best_fitness = -1
    best_v1_dist = 0
    best_v2_dist = 0

    no_improve_count = 0

    for gen in range(generations):
        fitness_results = [calculate_fitness(chrom, distances) for chrom in population]
        fitnesses = [res[0] for res in fitness_results]

        improved = False
        for i, fitness in enumerate(fitnesses):
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = population[i][:]
                _, best_v1_dist, best_v2_dist = fitness_results[i]
                improved = True

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1

        if target_fitness is not None and best_fitness >= target_fitness:
            break
        if patience is not None and no_improve_count >= patience:
            break

        new_population = [best_solution[:]]

        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)

            c1, c2 = single_point_crossover(p1, p2)

            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)

            new_population.extend([c1, c2])

        population = new_population[:pop_size]

    return best_solution, best_v1_dist, best_v2_dist

def plot_environment(grid, depot, points, best_solution):
    """
    Plots the grid with obstacles, the depot, and all delivery points.
    Green = Vehicle 1 (chromosome gene = 0)
    Orange = Vehicle 2 (chromosome gene = 1)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(grid, cmap='Greys', origin='upper')
    ax.scatter(depot[1], depot[0], c='blue', s=200, marker='D', zorder=5, label='Depot')
    v1_labeled = False
    v2_labeled = False

    for i, (r, c) in enumerate(points):
        if best_solution[i] == 0:
            color = 'green'
            label = 'Vehicle 1' if not v1_labeled else None
            v1_labeled = True
        else:
            color = 'orange'
            label = 'Vehicle 2' if not v2_labeled else None
            v2_labeled = True

        ax.scatter(c, r, c=color, s=150, edgecolors='black', zorder=5, label=label)
        ax.text(c + 0.2, r - 0.2, f'P{i+1}', fontsize=12, fontweight='bold', zorder=6)

    ax.set_title("Delivery Allocation Map\n(Green = Vehicle 1, Orange = Vehicle 2)", fontsize=14)
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def main():
    print("=" * 50)
    print("   Delivery Allocation — Load Balancing Project")
    print("=" * 50)
    grid = create_environment()
    depot = (0, 5)
    points = [(1, 1), (8, 1), (8, 5), (8, 8), (1, 8), (5, 9)]
    print("\n[STEP 1] Calculating shortest distances using BFS...")
    distances = []
    for i, p in enumerate(points):
        dist = bfs_shortest_path(grid, depot, p)
        distances.append(dist)
        print(f"  Depot -> P{i+1} {p} : {dist} steps")

    unreachable = [i for i, d in enumerate(distances) if not np.isfinite(d)]
    if unreachable:
        labels = ", ".join([f"P{i+1}" for i in unreachable])
        print(f"\nERROR: Unreachable delivery points: {labels}.")
        print("Fix the grid or point coordinates, then try again.")
        return

    print(f"\n  Total combined distance: {sum(distances)} steps")
    print(f"  Ideal per-vehicle share: {sum(distances) / 2} steps")
    print("\n[STEP 2] Running Genetic Algorithm (pop=20, gen=50, mutation=0.1)...")
    best_chromosome, dist_v1, dist_v2 = genetic_algorithm(
        distances=distances,
        pop_size=20,
        generations=50,
        mutation_rate=0.1,
        patience=8,
        target_fitness=1.0
    )
    print("\n" + "=" * 50)
    print("   OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Best Chromosome     : {best_chromosome}")
    print(f"Vehicle 1 Packages  : {[f'P{i+1}' for i in range(6) if best_chromosome[i] == 0]}")
    print(f"Vehicle 2 Packages  : {[f'P{i+1}' for i in range(6) if best_chromosome[i] == 1]}")
    print(f"Vehicle 1 Distance  : {dist_v1} steps")
    print(f"Vehicle 2 Distance  : {dist_v2} steps")
    print(f"Imbalance (|V1-V2|) : {abs(dist_v1 - dist_v2)} steps")

    if abs(dist_v1 - dist_v2) <= 2:
        print("\nConclusion: Workload is HIGHLY BALANCED ✓")
    else:
        print("\nConclusion: Best possible balance found.")
    plot_environment(grid, depot, points, best_chromosome)


if __name__ == "__main__":
    main()