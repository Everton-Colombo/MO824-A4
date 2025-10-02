from .AbstractGA import AbstractGA
from GA.Solution import Solution
import random

class GA(AbstractGA):
    ''' Standard Genetic Algorithm implementation. '''
    def __init__(self, 
                 obj_function,
                 generations: int,
                 pop_size: int,
                 chromosome_size: int,
                 mutation_rate: float,
                 init_stg: str = "std",
                 mutation_stg: str = "std",
                 population_stg: str = "std"):
        super().__init__(obj_function, generations, pop_size, chromosome_size, mutation_rate, init_stg, population_stg, mutation_stg)

    def create_empty_solution(self):
        return []

    def decode(self, chromosome) -> Solution:
        ''' Decodes a chromosome into a solution. '''
        solution = Solution()
        solution.elements = set(i for i in range(len(chromosome)) if chromosome[i] == 1)
        solution.cost = self.obj_function.evaluate(solution)
        return solution
    
    def generate_random_chromosome(self) -> list[int]:
        ''' Generates a random chromosome. '''
        chromosome = [0] * self.chromosome_size
        indices = random.sample(range(self.chromosome_size), self.chromosome_size // 2)
        for idx in indices:
            chromosome[idx] = 1
        return chromosome
    
    def fitness(self, chromosome) -> float:
        ''' Evaluates the fitness of a chromosome. '''
        return self.decode(chromosome).cost
    
    def mutate_gene(self, chromosome, locus: int):
        ''' Mutates a single gene in the chromosome. '''
        chromosome[locus] = 1 - chromosome[locus]  # Flip the bit

    