import os
import datetime

class ExperimentLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        
    def log(self, message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
        
        with open(self.log_file, "a") as f:
            f.write(formatted_message + "\n")

def get_logger(output_dir):
    log_file = os.path.join(output_dir, "experiment_logs.txt")
    return ExperimentLogger(log_file)
