import sys
import argparse

from pathlib import Path
from ortools.sat.python import cp_model
import pandas as pd
from util import validate_schedule, Schedule

tasks = ["A", "A_prime", "B", "C"]

def main():
    # Get CLI input
    parser = argparse.ArgumentParser(description="Create a call schedule")
    parser.add_argument(
        "schedule_file",
        help="The file containing vacation requests and fellow data",
        type=Path,
    )
    args = parser.parse_args()
    schedule = validate_schedule(args.schedule_file)

    # Initialize the model
    model = cp_model.CpModel()

    # Define solver vars
    # assignments[(p, w, t)] == 1 if person p does task t in week w
    assignments = {}
    for p in schedule.names:
        for w in schedule.weeks:
            for t in tasks:
                assignments[(p, w, t)] = model.NewBoolVar(f'p{p}_w{w}_t{t}')
    
    # Each week must have someone on A call and someone on C call
    for w in schedule.weeks:
        model.Add(
            sum(assignments[(p, w, 'A')] for p in schedule.names) == 1
        )
        model.Add(
            sum(assignments[(p, w, 'C')] for p in schedule.names) == 1
        )


if __name__ == "__main__":
    sys.exit(main())