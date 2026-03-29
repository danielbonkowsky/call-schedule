import sys

from ortools.sat.python import cp_model
from util import validate_schedule, parse_validate_args

def main():
    """Create a valid call schedule using constraint solvers"""

    args = parse_validate_args()
    schedule = validate_schedule(args)

    # Initialize the model
    model = cp_model.CpModel()

    # Define solver vars
    # assignments[(p, w, t)] == 1 if person p does task t in week w
    assignments = {}
    for p in schedule.names:
        for w in schedule.weeks:
            for t in schedule.tasks:
                assignments[(p, w, t)] = model.NewBoolVar(f"p{p}_w{w}_t{t}")
    
    # Week coverage -- each week must have someone on A call, someone on C call
    # and A' or B call depending on fellow
    for w in schedule.weeks:
        model.Add(
            sum(assignments[(p, w, "A")] for p in schedule.names) == 1
        )
        model.Add(
            sum(assignments[(p, w, "C")] for p in schedule.names) == 1
        )
        if schedule.week_has_fellow(w):
            model.Add(
                sum(assignments[(p, w, "B")] for p in schedule.names) == 1
            )
            model.Add(
                sum(assignments[(p, w, "A'")] for p in schedule.names) == 0
            )
        else:
            model.Add(
                sum(assignments[(p, w, "A'")] for p in schedule.names) == 1
            )
            model.Add(
                sum(assignments[(p, w, "B")] for p in schedule.names) == 0
            )
    
    # One task per person
    for p in schedule.names:
        for w in schedule.weeks:
            model.Add(sum(assignments[(p, w, t)] for t in schedule.tasks) <= 1)

    # Russ rules
    for w in schedule.weeks:
        model.Add(assignments[("Russ", w, "A'")] == 0)
        model.Add(assignments[("Russ", w, "B")] == 0)
        model.Add(assignments[("Russ", w, "C")] == 0)

    model.Add(sum(assignments[("Russ", w, "A")] for w in schedule.weeks) == 2)
    # TODO: Russ can only be on A call when there is a fellow


if __name__ == "__main__":
    sys.exit(main())