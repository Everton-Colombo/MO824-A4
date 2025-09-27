from scqbf_instance import *
from scqbf_solution import *
from scqbf_evaluator import *
import time

from typing import List


# Type aliases:
Chromosome = List[int]  # Example chromosome representation
Population = List[Chromosome]


class ScQbfGA:
    
    def __init__(self, instance: ScQbfInstance, debug_options: dict = {}):
        # GA related properties
        self.instance = instance
        self.evaluator = ScQbfEvaluator(instance)

        self.population: Population = []
        self.best_chromosome: Chromosome = None
        self.best_solution: ScQbfSolution = None
        
        self.debug_options = debug_options
        
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
            elif self.best_solution is not None and self.current_solution is not None and self._prev_best_solution is not None:
                if self.evaluator.evaluate_objfun(self.current_solution) <= self.evaluator.evaluate_objfun(self._prev_best_solution):
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
            self.history.append((self.evaluator.evaluate_objfun(self.best_solution) if self.best_solution else 0,
                                 self.evaluator.evaluate_objfun(self.current_solution) if self.current_solution else 0))


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
            new_population = self._select_population(offspring)
            
            for chromosome in new_population:
                fitness = self.evaluator.evaluate_objfun(self.docode(chromosome))
                if self.best_solution is None or fitness > self.evaluator.evaluate_objfun(self.best_solution):
                    self.best_solution = self.docode(chromosome)
                    self.best_chromosome = chromosome
        
        return self.best_solution
        
    def docode(self, chromosome: Chromosome) -> ScQbfSolution:
        pass
    
    def _initialize_population(self):
        pass
    
    def _select_parents(self) -> Population:
        pass
    
    def _crossover(self, parents: Population) -> Population:
        pass
    
    def _mutate(self, offspring: Population) -> Population:
        pass

    def _select_population(self, offspring: Population) -> Population:
        pass
    
    