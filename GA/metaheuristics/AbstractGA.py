from abc import ABC, abstractmethod
from ..problems import Evaluator
import random

class AbstractGA(ABC):

    
    ''' Abstract base class for Genetic Algorithm implementations. '''
    def __init__(self, 
                 obj_function: Evaluator,
                 generations: int,
                 pop_size: int,
                 chromosome_size: int,
                 mutation_rate: float,
                 init_stg: str = "std",
                 population_stg: str = "std",
                 mutation_stg: str = "std"):
        ''' Initializes the Genetic Algorithm with the given parameters. 
        Args:
            obj_function (Evaluator): The objective function evaluator.
            generations (int): Number of generations to run the algorithm.
            pop_size (int): Size of the population.
            chromosome_size (int): Size of each chromosome.
            mutation_rate (float): Mutation rate for the genetic algorithm.
            init_stg (str, optional): Strategy for initialization. Defaults to "std". Available: "std", "latin_hypercube".
            population_stg (str, optional): Strategy for population selection. Defaults to "std". Available: "std", "steady_state".
            mutation_stg (str, optional): Strategy for mutation. Defaults to "std". Available: "std", "adaptive".
        '''
        self.verbose = True

        self.obj_function = obj_function
        self.generations = generations
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.init_stg = init_stg    
        self.population_stg = population_stg
        self.mutation_stg = mutation_stg

        self.best_solution = None
        self.best_chromosome = None

    @abstractmethod
    def create_empty_solution(self):
        ''' Creates an empty solution structure. '''
        pass

    @abstractmethod
    def decode(self, chromosome):
        ''' Decodes a chromosome into a solution. '''
        pass

    @abstractmethod
    def generate_random_chromosome(self):
        ''' Generates a random chromosome. '''
        pass

    @abstractmethod
    def fitness(self, chromosome):
        ''' Evaluates the fitness of a chromosome. '''
        pass

    @abstractmethod
    def mutate_gene(self, chromosome, locus):
        ''' Mutates a single gene in the chromosome. '''
        pass

    @abstractmethod
    def repair_chromosome(self, chromosome):
        ''' Adjusts the chromosome to be a feasible solution. '''
        pass

    # ----------------------
    # Concrete Methods
    # ----------------------
    def initialize_population_std(self):
        ''' Initializes the population with random chromosomes using standard method. '''
        return [self.generate_random_chromosome() for _ in range(self.pop_size)]
    
    def initialize_population_latin_hypercube(self):
        ''' Initializes the population using Latin Hypercube Sampling. Assumes binary chromosomes. '''
        if self.pop_size % 2 != 0:
            print("[WARNING] Population size should be even for Latin Hypercube Sampling. Incrementing by 1.")
            self.pop_size += 1  # Adjust to the nearest even number
        
        self.population = []
        
        for i in range(self.pop_size):
            chromosome = [0] * self.obj_function.get_domain_size()
            self.population.append(chromosome)
        
        # For each gene position (column), create a random permutation
        for gene_pos in range(self.obj_function.get_domain_size()):
            # Create permutation of population indices [0, 1, 2, ..., pop_size-1]
            permutation = random.sample(range(self.pop_size), self.pop_size)
            
            # Assign alleles based on permutation index modulo 2
            for pop_idx in range(self.pop_size):
                allele = permutation[pop_idx] % 2
                self.population[pop_idx][gene_pos] = allele

    def initialize_population(self):
        ''' Initializes the population with random chromosomes. '''
        if self.init_stg == "std":
            return self.initialize_population_std()
        elif self.init_stg == "latin_hypercube":
            return self.initialize_population_latin_hypercube()
        else:
            raise ValueError(f"Unknown initialization strategy: {self.init_stg}")
    
    def get_best_chromosome(self, population):
        ''' Returns the best chromosome in the population based on fitness. '''
        best = max(population, key=self.fitness)
        return best
    
    def get_worst_chromosome(self, population):
        ''' Returns the worst chromosome in the population based on fitness. '''
        worst = min(population, key=self.fitness)
        return worst
    
    def crossover(self, parent1, parent2):
        ''' Performs single-point crossover between two parents to produce two offspring. '''
        point = random.randint(1, self.chromosome_size - 1)
        offspring1 = parent1[:point] + parent2[point:]
        offspring2 = parent2[:point] + parent1[point:]
        return offspring1, offspring2
    
    def mutate(self, chromosome):
        ''' Mutates the chromosome based on the mutation rate. '''
        for locus in range(self.chromosome_size):
            if random.random() < self.mutation_rate:
                self.mutate_gene(chromosome, locus)
        return chromosome
    
    def select_parents(self, population): #TODO Check if this is correct 
        ''' Standard tournament selection.  Selects the best out of 2 random individuals. '''
        parents = []
        while len(parents) < self.pop_size:
            tournament = random.sample(population, 2)
            winner = max(tournament, key=self.fitness)
            parents.append(winner)
        return parents

    def select_population_std(self, population, offspring):
        ''' Selects the next generation population using standard selection, retaining the best individuals. '''
        combined = population + offspring
        combined.sort(key=self.fitness, reverse=True)
        return combined[:self.pop_size]
    
    def select_population_steady_state(self, population, offspring):
        ''' Selects the next generation population using steady-state selection, replacing the worst individuals with new ones. '''
        population.sort(key=self.fitness, reverse=True)
        offspring.sort(key=self.fitness, reverse=True)

        for i in range(min(len(offspring), len(population))):
            population[-(i+1)] = offspring[i]

        return population

    def select_population(self, population, offspring):
        ''' Selects the next generation population based on the selected strategy. '''
        if self.population_stg == "std":
            return self.select_population_std(population, offspring)
        elif self.population_stg == "steady_state":
            return self.select_population_steady_state(population, offspring)

    def stopping_criteria(self, generation):
        ''' Checks if the stopping criteria are met. '''
        return generation >= self.generations
    
    def crossover_criteria(self):
        ''' Determines if crossover should occur. '''
        return True  # Always crossover in this simple implementation
    
    def get_diversity(self, population):
        ''' Calculates the normalized diversity of the population.'''
        P = len(population)

        if P == 0:
            return 0.0
        L = self.obj_function.get_domain_size()

        diversity_sum = 0.0
        for locus in range(L):
            ones = sum(ch[locus] for ch in population)  # count of 1s at locus
            p = ones / P
            # variance of Bernoulli = p*(1-p), max = 0.25 when p=0.5
            diversity_sum += p * (1 - p)
        # normalize by maximum possible sum (L * 0.25)
        return (diversity_sum / (0.25 * L))


    def mutation_criteria(self, population, generation):
        ''' Determines if mutation should occur. '''
        if self.mutation_stg == "adaptive" and generation % 5 == 0:
            # Adjust mutation rate based on population diversity every 5 generations
            if self.get_diversity(population) < 0.3:
                print("Low diversity detected, increasing mutation rate.", end="")
                self.mutation_rate = min(1.0, self.mutation_rate * 1.5)  # Increase mutation rate
            elif self.get_diversity(population) > 0.5:
                print("High diversity detected, decreasing mutation rate.", end="")
                self.mutation_rate = max(0.01, self.mutation_rate * 0.75)  # Decrease mutation rate
        elif self.mutation_stg == "always": # Always mutate
            return True
        elif self.mutation_stg == "never": # Never mutate
            return False
        # Standard random mutation based on mutation rate
        return random.random() < self.mutation_rate
        
    def solve(self):
        ''' Main method to run the Genetic Algorithm. '''
        population = self.initialize_population()
        self.best_chromosome = self.get_best_chromosome(population)
        self.best_solution = self.decode(self.best_chromosome)
        self.best_solution.cost = self.obj_function.evaluate(self.best_solution)
        
        generation = 0
        while not self.stopping_criteria(generation):
            # if the crossover criteria is met, perform crossover and generate new population
            if self.verbose:
                print(f"\nGeneration {generation}| ", end="")
            if self.crossover_criteria():
                parents = self.select_parents(population)
                offspring = []
                for i in range(0, len(parents), 2):
                    if i + 1 < len(parents):
                        off1, off2 = self.crossover(parents[i], parents[i + 1])
                        offspring.append(off1)
                        offspring.append(off2)
                population = self.select_population(population, offspring)

            # If mutation criteria is met, mutate the population
            if self.mutation_criteria(population, generation):
                population = [self.mutate(chrom) for chrom in population]
            
            self.best_chromosome = self.get_best_chromosome(population)
            current_best_solution = self.decode(self.best_chromosome)
            if current_best_solution.cost > self.best_solution.cost:
                self.best_solution = current_best_solution.copy()
                if self.verbose:
                    print(f"New Best {self.best_solution}", end="")
            generation += 1
        return self.best_solution