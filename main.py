import sys

from ortools.sat.python import cp_model
from util import parse_validate_args, build_schedule, solution_to_dataframe


A_CLOSE_PENALTY = 7               # penalty for scheduling someone to A call twice with less than two weeks between
NO_CALL_PENALTY = 7               # penalty for scheduling someone during a week they requested "no call"
NO_A_PENALTY = 7                  # penalty for scheduling someone during a week they requested "no a"
IF_NEEDED_PENALTY = 3             # penalty for scheduling someone during a week they requested "if needed"
TASK_COUNT_DEVIATION_PENALTY = 20 # penalty for deviating from someone's task count (per deviation)
ISOLATED_C_STREAK_PENALTY = 3     # penalty for scheduling C call in a one week block, rather than two weeks
PAIR_C_STREAK_PENALTY = 1         # penalty for scheduling C call in a pair rather than three weeks


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
    # and B' or B call depending on fellow
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
                sum(assignments[(p, w, "B'")] for p in schedule.names) == 0
            )
        else:
            model.Add(
                sum(assignments[(p, w, "B'")] for p in schedule.names) == 1
            )
            model.Add(
                sum(assignments[(p, w, "B")] for p in schedule.names) == 0
            )
    
    # Max of one task per person per week
    for p in schedule.names:
        for w in schedule.weeks:
            model.Add(sum(assignments[(p, w, t)] for t in schedule.tasks) <= 1)
    
    # Vacation constraints and preferences
    vacation_penalties = []
    for p in schedule.names:
        for w in schedule.weeks:
            status = schedule.status_during_week(p, w)
            if status == "vacation":
                for t in schedule.tasks:
                    model.Add(assignments[(p, w, t)] == 0)
            elif status == "no call":
                vacation_penalties.append(
                    NO_CALL_PENALTY * 
                    sum(assignments[(p, w, t)] for t in schedule.tasks)
                )
            elif status == "if needed":
                vacation_penalties.append(
                    IF_NEEDED_PENALTY * 
                    sum(assignments[(p, w, t)] for t in schedule.tasks)
                )
            elif status == "no a":
                vacation_penalties.append(
                    NO_A_PENALTY * assignments[(p, w, "A")]
                )

    # A person cannot be assigned A Call 2 weeks in a row
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 1):
            w1, w2 = schedule.weeks[i], schedule.weeks[i + 1]
            model.Add(
                assignments[(p, w1, "A")] + assignments[(p, w2, "A")] <= 1
            )

    # Prefer > 2 weeks between A Call assignments (soft constraint)
    a_close_penalties = []
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 2):
            w1, w3 = schedule.weeks[i], schedule.weeks[i + 2]
            close = model.NewBoolVar(f"a_close_{p}_{i}")
            model.AddMinEquality(
                close,
                [assignments[(p, w1, "A")],
                 assignments[(p, w3, "A")]]
            )
            a_close_penalties.append(A_CLOSE_PENALTY * close)

    # B', B, and C call can be assigned up to 2 weeks in a row
    # Exception: persons with > 10 C assignments may be assigned C 3 weeks in a row
    for p in schedule.names:
        high_c = schedule.target_task_amount(p, "C") > 10
        for i in range(len(schedule.weeks) - 2):
            w1 = schedule.weeks[i]
            w2 = schedule.weeks[i + 1]
            w3 = schedule.weeks[i + 2]
            model.Add(
                assignments[(p, w1, "B'")]
                + assignments[(p, w2, "B'")]
                + assignments[(p, w3, "B'")]
                <= 2
            )
            model.Add(
                assignments[(p, w1, "B")]
                + assignments[(p, w2, "B")]
                + assignments[(p, w3, "B")]
                <= 2
            )
            if not high_c:
                model.Add(
                    assignments[(p, w1, "C")]
                    + assignments[(p, w2, "C")]
                    + assignments[(p, w3, "C")]
                    <= 2
                )

    # Prefer C call in 2-week blocks over 1-week blocks (soft constraint)
    # For persons with > 10 C assignments, 3-week blocks are slightly preferred over 2-week blocks
    c_streak_penalties = []
    for p in schedule.names:
        high_c = schedule.target_task_amount(p, "C") > 10
        for i in range(len(schedule.weeks) - 2):
            w1 = schedule.weeks[i]
            w2 = schedule.weeks[i + 1]
            w3 = schedule.weeks[i + 2]
            isolated = model.NewBoolVar(f"c_isolated_{p}_{i}")
            model.AddMinEquality(
                isolated,
                [
                    assignments[(p, w2, "C")],
                    assignments[(p, w1, "C")].Not(),
                    assignments[(p, w3, "C")].Not()
                ]
            )
            c_streak_penalties.append(ISOLATED_C_STREAK_PENALTY * isolated)
            if high_c:
                # Penalize pairs that don't extend into a triple (prefers triples over pairs)
                pair_no_ext = model.NewBoolVar(f"c_pair_no_ext_{p}_{i}")
                model.AddMinEquality(
                    pair_no_ext,
                    [
                        assignments[(p, w1, "C")],
                        assignments[(p, w2, "C")],
                        assignments[(p, w3, "C")].Not(),
                    ]
                )
                c_streak_penalties.append(PAIR_C_STREAK_PENALTY * pair_no_ext)

    # Can't give three assignments in a row (regardless of type)
    for p in schedule.names:
        for i in range(len(schedule.weeks) - 2):
            w1, w2, w3 = schedule.weeks[i]
            w2 = schedule.weeks[i + 1]
            w3 = schedule.weeks[i + 2]
            model.Add(
                sum(
                    assignments[(p, w, t)] 
                    for w in [w1, w2, w3] for t in schedule.tasks
                ) <= 2
            )

    # Can't assign call on BOTH 12/21 and 12/28 (within the same year)
    christmas_weeks = {
        w[6:]: w for w in schedule.weeks if w.startswith("12-21")
    }
    new_years_weeks = {
        w[6:]: w for w in schedule.weeks if w.startswith("12-28")
    }
    for year in set(christmas_weeks) & set(new_years_weeks):
        w_christmas = christmas_weeks[year]
        w_new_years = new_years_weeks[year]
        for p in schedule.names:
            model.Add(
                sum(assignments[(p, w_christmas, t)] for t in schedule.tasks)
                + sum(assignments[(p, w_new_years, t)] for t in schedule.tasks)
                <= 1
            )

    # Russ rules
    for w in schedule.weeks:
        model.Add(assignments[("Russ", w, "B'")] == 0)
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

    # Total assignments per person should be close to their target total (soft constraint)
    total_deviation_penalties = []
    max_weeks = len(schedule.weeks)
    for p in schedule.names:
        total_target = sum(
            schedule.target_task_amount(p, t) for t in schedule.tasks
        )
        total_assigned = sum(
            assignments[(p, w, t)] 
            for w in schedule.weeks for t in schedule.tasks
        )
        diff = model.NewIntVar(-max_weeks, max_weeks, f"total_diff_{p}")
        model.Add(diff == total_assigned - total_target)
        abs_diff = model.NewIntVar(0, max_weeks, f"total_abs_diff_{p}")
        model.AddAbsEquality(abs_diff, diff)
        total_deviation_penalties.append(
            TASK_COUNT_DEVIATION_PENALTY * abs_diff
        )

    model.Minimize(
        sum(a_close_penalties) 
        + sum(c_streak_penalties) 
        + sum(total_deviation_penalties) 
        + sum(vacation_penalties)
    )

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
