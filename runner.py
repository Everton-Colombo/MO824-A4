from scqbf.scqbf_instance import *
from scqbf.scqbf_solution import *
from scqbf.scqbf_evaluator import *
from scqbf.scqbf_ga import *
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_instance(file, log_dir, pop_size, mutation_rate_mult, ga_strategy, termination_options, debug_options):
    print(f"[DEBUG] run_instance called with file: {file}", flush=True)
    in_file = os.path.basename(file).replace(".txt", "")
    log_file = os.path.join(log_dir, f"{in_file}.log")
    filename = os.path.splitext(os.path.basename(file))[0]

    if not os.path.exists(log_dir):
        raise ValueError(f"Log directory {log_dir} does not exist.")

    print(f"[{time.strftime('%H:%M')}] Running instance {filename}...", flush=True)
    
    start = time.time()
    try:
        with open(file, "r") as fin, open(log_file, "w") as fout:
            instance = read_max_sc_qbf_instance(file)
            ga = ScQbfGA(instance, 
            population_size=pop_size, 
            mutation_rate_multiplier=mutation_rate_mult, 
            ga_strategy=ga_strategy, 
            debug_options=debug_options, 
            termination_options=termination_options)
            
            solution = ga.solve()
            print(f"[DEBUG] Solution obtained: {solution}")
            fout.write(f"Solution: {solution}\n")

        end = time.time()
        minutes, seconds = divmod(end - start, 60)
        with open(log_file, "a") as fout:
            fout.write(f"\n--- Finished in {end - start:.2f} seconds ---\n")

        print(f"[{time.strftime('%H:%M')}] Finished {filename} in {minutes:.0f}m{seconds:.2f}s\n", flush=True)
        return filename, True

    except Exception as e:
        with open(log_file, "a") as fout:
            fout.write(f"\n--- Error: {e} ---\n")
        print(f" Error running {filename}: {e}", flush=True)
        return filename, False
    
def run_experiment(log_dir, pop_size, mutation_rate_mult, GA_strategy=GAStrategy(), debug_options=None, termination_options=None):
    # prepare in directory
    log_dir = "logs/" + log_dir
    os.makedirs(log_dir, exist_ok=True)
    for f in glob.glob(os.path.join(log_dir, "*")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"[WARNING] Could not remove {f}: {e}")

    files = sorted(glob.glob(os.path.join("instances", "*.txt")))

    # Debug: Print the files list
    print(f"[DEBUG] Files to process: {files}")
    file = files[0] if files else None

    # number of workers = available threads or number of instances
    max_workers = min(len(files), os.cpu_count() or 4)

    print(f"Running {len(files)} instances in up to {max_workers} processes...\n")

    # runs in parallel
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            print("[DEBUG] Submitting tasks to ProcessPoolExecutor...")
            futures = [
                executor.submit(
                    run_instance,
                    file,
                    log_dir,
                    pop_size,
                    mutation_rate_mult,
                    GA_strategy,
                    termination_options,
                    debug_options
                )
            ]

            print("[DEBUG] Tasks submitted. Waiting for completion...")
            for future in as_completed(futures):
                try:
                    result = future.result()
                    print(f"[DEBUG] Completed: {result}")
                except Exception as e:
                    print(f"[ERROR] Exception in worker: {e}")
    except Exception as e:
        print(f"Error in ProcessPoolExecutor: {e}")
        return

    print("All files processed. Check the logs directory for outputs.")

if __name__ == "__main__":
    # Example usage
   run_experiment(log_dir="test_experiment", 
               pop_size=100,
               mutation_rate_mult=2,
               GA_strategy=GAStrategy(population_init="random", mutation_strategy="standard"),
               termination_options={"max_iter": 1},
               debug_options={"verbose": False}
               )