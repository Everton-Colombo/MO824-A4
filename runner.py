import subprocess
import time
import os
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

minutes = 0.5  # timeout em minutos

def run_instance(file, t_out, log_dir):
    base = os.path.basename(file).replace(".txt", "")
    log_file = os.path.join(log_dir, f"{base}.log")
    filename = os.path.splitext(os.path.basename(file))[0]

    print(f"[{time.strftime('%H:%M')}] Running main.py with input {filename}...")

    start = time.time()
    try:
        with open(file, "r") as fin, open(log_file, "w") as fout:
            subprocess.run(
                ["python", "main.py"],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.STDOUT,
                timeout=t_out
            )
        end = time.time()

        with open(log_file, "a") as fout:
            fout.write(f"\n--- Finished in {end - start:.2f} seconds ---\n")

        print(f"Finished {filename} in {end - start:.2f} seconds\n")
        return filename, True

    except subprocess.TimeoutExpired:
        with open(log_file, "a") as fout:
            fout.write(f"\n--- Execution timed out after {t_out} seconds ---\n")
        print(f"⏱ Timeout: {filename}")
        return filename, False

    except Exception as e:
        with open(log_file, "a") as fout:
            fout.write(f"\n--- Error: {e} ---\n")
        print(f"❌ Error running {filename}: {e}")
        return filename, False


def main():
    input_dir = "instances"
    log_dir = "logs"

    # prepara diretório de logs
    os.makedirs(log_dir, exist_ok=True)
    for f in glob.glob(os.path.join(log_dir, "*")):
        try:
            os.remove(f)
        except Exception as e:
            print(f"Warning: Could not remove {f}: {e}")

    t_out = 60 * minutes
    files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))

    # número de processos = núcleos disponíveis ou quantidade de instâncias
    max_workers = min(len(files), os.cpu_count() or 4)

    print(f"👉 Executando {len(files)} instâncias em até {max_workers} processos...\n")

    # executa em paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_instance, file, t_out, log_dir) for file in files]

        # acompanhar progresso conforme terminam
        for future in as_completed(futures):
            filename, success = future.result()
            if success:
                print(f"✅ Concluído: {filename}")
            else:
                print(f"⚠️ Falhou ou expirou: {filename}")

    print("All files processed. Check the logs directory for outputs.")


if __name__ == "__main__":
    main()
