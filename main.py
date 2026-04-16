import sys

from ortools.sat.python import cp_model
from util import parse_validate_args, build_schedule, solution_to_dataframe

def main() -> int:
    """Create a valid call schedule using constraint solvers"""

    args = parse_validate_args()
    schedule = build_schedule(args)
    
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
    
    # Max of one task per person per week
    for p in schedule.names:
        for w in schedule.weeks:
            model.Add(sum(assignments[(p, w, t)] for t in schedule.tasks) <= 1)

    # A person cannot be assigned A Call 2 weeks in a row
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 1):
            w1, w2 = schedule.weeks[i], schedule.weeks[i + 1]
            model.Add(assignments[(p, w1, "A")] + assignments[(p, w2, "A")] <= 1)

    # Prefer > 2 weeks between A Call assignments (soft constraint)
    a_close_penalties = []
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 2):
            w1, w3 = schedule.weeks[i], schedule.weeks[i + 2]
            close = model.NewBoolVar(f"a_close_{p}_{i}")
            model.AddMinEquality(close, [assignments[(p, w1, "A")], assignments[(p, w3, "A")]])
            a_close_penalties.append(close)

    # A’ or B call can be assigned up to 2 weeks in a row
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 2):
            w1, w2, w3 = schedule.weeks[i], schedule.weeks[i + 1], schedule.weeks[i + 2]
            model.Add(
                assignments[(p, w1, "A'")] + assignments[(p, w2, "A'")] + assignments[(p, w3, "A'")] <= 2
            )
            model.Add(
                assignments[(p, w1, "B")] + assignments[(p, w2, "B")] + assignments[(p, w3, "B")] <= 2
            )

    # C call can be assigned up to 2 weeks in a row (soft constraint -- preferred but not required)
    c_streak_penalties = []
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 2):
            w1, w2, w3 = schedule.weeks[i], schedule.weeks[i + 1], schedule.weeks[i + 2]
            streak = model.NewBoolVar(f"c_streak_{p}_{i}")
            model.AddMinEquality(
                streak,
                [assignments[(p, w1, "C")], assignments[(p, w2, "C")], assignments[(p, w3, "C")]]
            )
            c_streak_penalties.append(streak)

    # Russ rules
    for w in schedule.weeks:
        model.Add(assignments[("Russ", w, "A'")] == 0)
        model.Add(assignments[("Russ", w, "B")] == 0)
        model.Add(assignments[("Russ", w, "C")] == 0)

    model.Add(sum(assignments[("Russ", w, "A")] for w in schedule.weeks) == 2)

    # Russ can only be on A call when there is a fellow
    for w in schedule.weeks:
        if not schedule.week_has_fellow(w):
            model.Add(assignments[("Russ", w, "A")] == 0)

    # If someone has 0 instances of a task in the Assignments table, they cannot be assigned that task
    for p in schedule.names:
        for t in schedule.tasks:
            if schedule.target_task_amount(p, t) == 0:
                for w in schedule.weeks:
                    model.Add(assignments[(p, w, t)] == 0)

    # Total assignments per person should not vary by more than +/-1 from their target total
    for p in schedule.names:
        total_target = sum(schedule.target_task_amount(p, t) for t in schedule.tasks)
        total_assigned = sum(assignments[(p, w, t)] for w in schedule.weeks for t in schedule.tasks)
        model.Add(total_assigned >= total_target - 1)
        model.Add(total_assigned <= total_target + 1)

    model.Minimize(sum(a_close_penalties) + sum(c_streak_penalties))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        print("Optimal solution found")
    elif status == cp_model.FEASIBLE:
        print("Feasible solution found (but not proven optimal)")
    elif status == cp_model.INFEASIBLE:
        print("No solution exists")
        return -1
    elif status == cp_model.MODEL_INVALID:
        print("Model is invalid")
        return -1
    else:  # UNKNOWN
        print("Solver stopped before finding a solution")
        return -1

    df = solution_to_dataframe(solver, assignments, schedule)
    df.to_csv(args.output_file, index=False)
    print(f"Schedule written to {args.output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
