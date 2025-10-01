from .scqbf_instance import *
from .scqbf_solution import *
from .scqbf_evaluator import *
import time
import random
from dataclasses import dataclass

from typing import List, Literal


# Type aliases:
Chromosome = List[int]  # Example chromosome representation
Population = List[Chromosome]


@dataclass
class GAStrategy:
    population_init: Literal["random", "latin_hypercube"] = "random"
    # Add other strategies as needed
class ScQbfGA:
    
    def __init__(self, instance: ScQbfInstance, population_size: int = 100, mutation_rate_multiplier: int = 1,
                 ga_strategy: GAStrategy = GAStrategy(), debug_options: dict = {}):
        # GA related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)
        self.population: Population = []
        self.best_chromosome: Chromosome = None
        self.best_solution: ScQbfSolution = None
        self.population_size = population_size
        self.mutation_rate_multiplier = mutation_rate_multiplier

        self.ga_strategy = ga_strategy
        self.debug_options = debug_options
        self._cache = {}
        
        # Internal properties for managing execution and termination criteria
        self._start_time = None                         # Start time of the algorithm
        self.execution_time = 0.0                       # Total execution time
        self._iter = 0                                  # Current iteration
        self._no_improvement_iter = 0                   # Iterations since last improvement
        self._prev_best_solution = None                 # Previous best solution to track improvements
        self.stop_reason = None                         # Reason for stopping the algorithm (e.g., max_iter, time_limit, etc.)
        self.history: list[tuple[float, float]] = []    # History of best and current solutions' objective values
        

    def _eval_termination_condition(self) -> bool:
        """ Check if the termination condition is met, while also managing termination criteria properties."""

        self._iter += 1
        self.execution_time = time.time() - self._start_time
        
        max_iter = self.debug_options.get("max_iter", None)
        time_limit_secs = self.debug_options.get("time_limit_secs", None)
        patience = self.debug_options.get("patience", None)

        if max_iter is not None and self._iter >= max_iter:
            self.stop_reason = "max_iter"
            return True
        if time_limit_secs is not None and self.execution_time >= time_limit_secs:
            self.stop_reason = "time_limit"
            return True
        if patience is not None:
            if self._no_improvement_iter >= patience:
                self.stop_reason = "patience_exceeded"
                return True
            elif self.best_solution is not None and self._prev_best_solution is not None:
                if self.evaluator.evaluate_objfun(self.best_solution) <= self.evaluator.evaluate_objfun(self._prev_best_solution):
                    self._no_improvement_iter += 1
                else:
                    self._no_improvement_iter = 0

        self._prev_best_solution = self.best_solution

        return False
    
    def _perform_debug_actions(self):
        """ Perform debug actions, such as logging or printing debug information. """
        if self.debug_options.get("verbose", False):
            print(f"Iteration {self._iter}: Best fitness = {self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 'N/A'}")

        if self.debug_options.get("save_history", False):
            self.history.append((self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0))


    def solve(self) -> ScQbfSolution:
        """ Main method to solve the problem using a genetic algorithm. """
        self._initialize_population()
        
        self._start_time = time.time()
        self._iter = 0
        while not self._eval_termination_condition():
            self._perform_debug_actions()
            
            parents = self._select_parents()
            offspring = self._crossover(parents)
            offspring = self._mutate(offspring)
            self.population = self._select_population(offspring)

            for chromosome in self.population:
                solution, fitness = self._cached_decode_and_eval(chromosome)
                if self.best_solution is None or fitness > self.evaluator.evaluate_objfun(self.best_solution):
                    self.best_solution = solution
                    self.best_chromosome = chromosome
        
        return self.best_solution
        
    def decode(self, chromosome: Chromosome) -> ScQbfSolution:
        """ Decode a chromosome into a ScQbfSolution. """
        solution = ScQbfSolution([])
        for i in range(len(chromosome)):
            if chromosome[i] == 1:
                solution.elements.append(i)

        return solution

    def _cached_decode_and_eval(self, chromosome: Chromosome) -> tuple[ScQbfSolution, float]:
        key = tuple(chromosome)  # O(n) to create tuple
        if key not in self._cache:  # O(1) average lookup
            solution = self.decode(chromosome)
            fitness = self.evaluator.evaluate_objfun(solution)
            self._cache[key] = (solution, fitness)  # O(1) average insertion
        return self._cache[key]

    def _initialize_population(self):
        if self.ga_strategy.population_init == "random":
            self._initialize_population_random()
        elif self.ga_strategy.population_init == "latin_hypercube":
            self._initialize_population_latin_hypercube()
        else:
            raise ValueError(f"Unknown population initialization strategy: {self.ga_strategy.population_init}")
        
    def _initialize_population_random(self):
        self.population = []
        for _ in range(self.population_size):
            chromosome = [0] * self.instance.n
            num_ones = random.randint(1, self.instance.n)
            ones_indices = random.sample(range(self.instance.n), num_ones)
            for idx in ones_indices:
                chromosome[idx] = 1
            chromosome = self._make_feasible(chromosome)
            self.population.append(chromosome)

    def _initialize_population_latin_hypercube(self):
        self.population = []
        # Implement Latin Hypercube Sampling initialization here
        # This is a placeholder for the actual implementation
        raise NotImplementedError("Latin Hypercube initialization not implemented yet.")

    def _make_feasible(self, chromosome: Chromosome) -> Chromosome:
        """
        If the chromosome is not feasible, add random elements that improve coverage until it becomes feasible.
        """
        
        decoded_solution, _ = self._cached_decode_and_eval(chromosome)
        while not self.evaluator.is_solution_feasible(decoded_solution):
            cl = [i for i in range(self.instance.n) if chromosome[i] == 0 and self.evaluator.evaluate_insertion_delta_coverage(i, decoded_solution) > 0]
            chosen = random.choice(cl)
            chromosome[chosen] = 1
            decoded_solution.elements.append(chosen)
            
        return chromosome

    def _select_parents(self) -> Population:
        """ Tournament selection implementation."""
        parents: Population = []
        
        while len(parents) < self.population_size:
            tournament = random.sample(self.population, 2)
            tournament_fitness = [self._cached_decode_and_eval(chrom)[1] for chrom in tournament]
            winner = tournament[tournament_fitness.index(max(tournament_fitness))]
            parents.append(winner)
        
        return parents
    
    def _crossover(self, parents: Population) -> Population:
        """
        Two-point crossover implementation.
        """
        offspring: Population = []
        
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[i + 1]
            
            point1, point2 = sorted([
                random.randint(0, self.instance.n - 1),
                random.randint(0, self.instance.n - 1)
            ])
            
            offspring1 = parent1[:point1] + parent2[point1:point2] + parent1[point2:]
            offspring2 = parent2[:point1] + parent1[point1:point2] + parent2[point2:]

            offspring1 = self._make_feasible(offspring1)
            offspring2 = self._make_feasible(offspring2)

            offspring.extend([offspring1, offspring2])
        
        return offspring
    
    def _mutate(self, offspring: Population) -> Population:
        for chromosome in offspring:
            mutation_rate = 1 / self.instance.n * self.mutation_rate_multiplier # when multiplier = 1, expected 1 mutation per chromosome
            for i in range(len(chromosome)):
                if random.random() < mutation_rate:
                    chromosome[i] = 1 - chromosome[i]  # Flip bit
            chromosome = self._make_feasible(chromosome)
        
        return offspring

    def _select_population(self, offspring: Population) -> Population:
        """Elitist selection implementation. replace worst single offspring with best from previous generation."""
        worst_chromosome = min(offspring, 
                            key=lambda chrom: self._cached_decode_and_eval(chrom)[1])
        
        # Only replace if the worst offspring is worse than the best from previous generation
        if (self.best_chromosome is not None and 
            self._cached_decode_and_eval(worst_chromosome)[1] < 
            self._cached_decode_and_eval(self.best_chromosome)[1]):
            
            worst_index = offspring.index(worst_chromosome)
            offspring[worst_index] = self.best_chromosome
        
        return offspring

