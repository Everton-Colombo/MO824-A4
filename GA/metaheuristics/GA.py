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

    def repair_chromosome(self, chromosome):
        ''' Repairs a chromosome to ensure it represents a feasible solution. 
        Greedy approach to ensure all elements are covered.
        '''
        # 1. Identify covered elements
        solution = self.decode(chromosome)
        covered = self.obj_function.coverage(solution)
        elements = set(range(self.obj_function.get_domain_size()))

        # 2. While there are still uncovered elements
        while len(covered) != self.obj_function.get_domain_size():
            uncovered = elements - covered

            # 2a. Evaluate candidates: subsets that are not active yet
            best_idx = None
            best_ratio = -1
            for i, active in enumerate(chromosome):
                if active == 0:
                    new_covered = uncovered & set(self.obj_function.sets[i])
                    if len(new_covered) > 0:
                        ratio = len(new_covered) * self.obj_function.evaluate_insertion_cost(i, solution)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_idx = i

            # 2b. Activate the best subset found
            chromosome[best_idx] = 1
            covered.update(self.obj_function.sets[best_idx])

    
    def fitness(self, chromosome) -> float:
        ''' Evaluates the fitness of a chromosome. '''
        solution = self.decode(chromosome)
        # If the chromosome is not feasible but promising, repair it
        if self.best_solution is not None and solution.cost >= self.best_solution.cost * 1.1:
            self.repair_chromosome(chromosome)
        return self.decode(chromosome).cost
    
    def mutate_gene(self, chromosome, locus: int):
        ''' Mutates a single gene in the chromosome. '''
        chromosome[locus] = 1 - chromosome[locus]  # Flip the bit

    
