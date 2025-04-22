import os
import subprocess
import time

TASKS_DIR = "tasks/"
LOG = "memory/learning_logs/bateria_autoavaliacao.log"

def run_task(task_file):
    cmd = [
        "python", "-m", "a3x.assistant_cli",
        "--task", f"{task_file}"
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    end = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(f"=== {task_file} ===\n")
        f.write(f"Start: {start}, End: {end}, Duration: {end-start:.2f}s\n")
        f.write(result.stdout)
        f.write(result.stderr)
        f.write("\n\n")
    print(f"Task {task_file} executed. Status: {result.returncode}")

def main():
    for fname in os.listdir(TASKS_DIR):
        if fname.endswith(".json"):
            run_task(os.path.join(TASKS_DIR, fname))

if __name__ == "__main__":
    main()