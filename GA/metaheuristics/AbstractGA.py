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
                 strategy: str = "std",
                 population_stg: str = "std"):
        ''' Initializes the Genetic Algorithm with the given parameters. 
        Args:
            obj_function (Evaluator): The objective function evaluator.
            generations (int): Number of generations to run the algorithm.
            pop_size (int): Size of the population.
            chromosome_size (int): Size of each chromosome.
            mutation_rate (float): Mutation rate for the genetic algorithm.
            strategy (str, optional): Strategy for selection. Defaults to "std".
            population_stg (str, optional): Strategy for population selection. Defaults to "std".
        '''
        self.verbose = True

        self.obj_function = obj_function
        self.generations = generations
        self.pop_size = pop_size
        self.chromosome_size = chromosome_size
        self.mutation_rate = mutation_rate
        self.strategy = strategy    
        self.population_stg = population_stg

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

    # ----------------------
    # Concrete Methods
    # ----------------------
    def initialize_population(self):
        ''' Initializes the population with random chromosomes. '''
        population = [self.generate_random_chromosome() for _ in range(self.pop_size)]
        return population
    
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
    
    def select_parents(self, population):
        ''' Selects parents for crossover based on the selection strategy. '''
        parents = []
        if self.strategy == "std":
            ''' Standard tournament selection.  Selects the best out of 2 random individuals. '''
            while len(parents) < self.pop_size:
                tournament = random.sample(population, 2)
                winner = max(tournament, key=self.fitness)
                parents.append(winner)
        else:
            raise ValueError(f"Unknown selection strategy: {self.strategy}")
        return parents
    
    def select_population_std(self, population, offspring):
        ''' Selects the next generation population using standard selection. '''
        combined = population + offspring
        combined.sort(key=self.fitness, reverse=True)
        return combined[:self.pop_size]

    def select_population(self, population, offspring):
        ''' Selects the next generation population based on the selected strategy. '''
        if self.population_stg == "std":
            return self.select_population_std(population, offspring)

    def solve(self):
        ''' Main method to run the Genetic Algorithm. '''
        population = self.initialize_population()
        self.best_chromosome = self.get_best_chromosome(population)
        self.best_solution = self.decode(self.best_chromosome)
        
        for generation in range(self.generations):
            parents = self.select_parents(population)
            offspring = []
            for i in range(0, self.pop_size, 2):
                parent1 = parents[i]
                parent2 = parents[i+1] if i+1 < self.pop_size else parents[0]
                child1, child2 = self.crossover(parent1, parent2)
                offspring.append(self.mutate(child1))
                if len(offspring) < self.pop_size:
                    offspring.append(self.mutate(child2))
            population = self.select_population(population, offspring)
            best_in_gen = self.get_best_chromosome(population)
            if self.best_chromosome is None or self.fitness(best_in_gen) > self.fitness(self.best_chromosome):
                self.best_chromosome = best_in_gen
                self.best_solution = self.decode(best_in_gen)
                if self.verbose:
                    print(f"Generation {generation+1}: {self.best_solution}")
        return self.best_solution