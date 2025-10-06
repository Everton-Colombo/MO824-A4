from scqbf.scqbf_instance import *
from scqbf.scqbf_solution import *
from scqbf.scqbf_evaluator import *
from scqbf.scqbf_ga import *

def run_single_experiment(args):
    """Helper function to run a single experiment"""
    instance_path, gen, inst, exp_num, config, pop_size, mutation_rate = args
    
    try:
        print(f"Running experiment {exp_num} with instance {instance_path}")
        instance = read_max_sc_qbf_instance(instance_path)
        
        time_limit = 60 * 30
        ga = ScQbfGA(instance, pop_size, mutation_rate, termination_options={'time_limit_secs': time_limit, 'patience': 1000}, ga_strategy=config)
        best_solution = ga.solve()

        evaluator = ScQbfEvaluator(instance)
        result = {
            'gen': gen,
            'inst': inst,
            'n': instance.n,
            'stop_reason': ga.stop_reason,
            'best_objective': evaluator.evaluate_objfun(best_solution),
            'coverage': evaluator.evaluate_coverage(best_solution),
            'time_taken': ga.execution_time,
            'exp_num': exp_num,
            'instance_path': instance_path,
            'selected_elements': best_solution.elements
        }
        
        print(f"Experiment {exp_num} completed successfully")
        return result
    except Exception as e:
        print(f"Experiment {exp_num} failed with error: {e}")
        raise
