import sys
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import argparse


class Schedule:
    """Schedule class with validated dataframe for easy access"""

    def __init__(
            self, 
            vacation: pd.DataFrame,
            fellow_schedule: pd.DataFrame,
            task_counts: pd.DataFrame,
        ):
        self._vacation = vacation
        self._fellow_schedule = fellow_schedule
        self._task_counts = task_counts

    @property
    def names(self) -> list[str]:
        """Returns the names of all physicians in the schedule"""

        return list(self._vacation.columns)[1:]
    
    @property
    def weeks(self) -> list[str]:
        """Returns a list of all the weeks in the schedule"""

        return self._vacation["Week"].to_list()
    
    @property
    def tasks(self) -> list[str]:
        """Returns all the tasks defined in task_counts"""

        return self._task_counts["Task"].to_list()
    
    def week_has_fellow(self, week: str) -> bool:
        """Determine whether a given week has a fellow assigned"""

        result = self._fellow_schedule.loc[self._fellow_schedule["Week"] == week, "Fellow"]
        return result.values[0] in {"", np.nan, None}


def _validate_weeks(
        vacation: pd.DataFrame, 
        fellow_schedule: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ensures that dataframes have matching weeks and fixes formatting """

    # Standardize date formatting as mm-dd-yy
    def parse_weeks(series: pd.Series, source: str) -> pd.Series:
        formatted = []
        for idx, val in series.items():
            try:
                formatted.append(pd.to_datetime(val).strftime("%m-%d-%y"))
            except Exception:
                sys.exit(f'{source}: Could not parse date "{val}" in row {idx + 2}')
        return pd.Series(formatted, index=series.index)

    vacation = vacation.copy()
    fellow_schedule = fellow_schedule.copy()
    vacation["Week"] = parse_weeks(vacation["Week"], "Vacation file")
    fellow_schedule["Week"] = parse_weeks(fellow_schedule["Week"], "Fellow schedule")

    # Ensure that weeks match between dataframes
    if vacation["Week"].tolist() != fellow_schedule["Week"].tolist():
        sys.exit("Weeks do not match between vacation schedule and fellow schedule")

    # Ensure that weeks start on Monday and are 7 days apart
    dates = [datetime.strptime(val, "%m-%d-%y") for val in vacation["Week"]]
    for i, dt in enumerate(dates):
        if dt.weekday() != 0:
            sys.exit(
                f'Vacation file: Week "{vacation["Week"].iloc[i]}" does not start on a Monday'
            )
        if i > 0:
            delta = (dt - dates[i - 1]).days
            if delta != 7:
                sys.exit(
                    f'Weeks "{vacation["Week"].iloc[i-1]}" and "{vacation["Week"].iloc[i]}" '
                    f'are {delta} days apart, expected 7'
                )

    return vacation, fellow_schedule


def validate_schedule(args: argparse.Namespace) -> Schedule:
    """Takes in validated args and returns a full Schedule"""

    # Read file
    vacation = pd.read_csv(args.vacation)
    fellow_schedule = pd.read_csv(args.fellow_schedule)
    task_counts = pd.read_csv(args.task_counts)

    # Validate file formats
    if list(vacation.columns)[0] != "Week":
        sys.exit(
            f'Vacation file: Expected first column "Week", got "{list(vacation.columns)[0]}"'
        )
    if list(fellow_schedule.columns)[0] != "Week":
        sys.exit(
            f'Fellow schedule: Expected first column "Week", got "{list(fellow_schedule.columns)[0]}"'
        )
    if list(fellow_schedule.columns)[1] != "Fellow":
        sys.exit(
            f'Fellow schedule: Expected second column "Fellow", got "{list(fellow_schedule.columns)[1]}"'
        )
    if len(list(fellow_schedule.columns)) > 2:
        sys.exit(
            f'Fellow schedule: Extra column {list(fellow_schedule.columns)[2]}'
        )

    vacation, fellow_schedule = _validate_weeks(vacation, fellow_schedule)

    # Make sure names match between vacation schedule and task counts
    vacation_names = set(vacation.columns[1:])
    task_count_names = set(task_counts.columns[1:])
    if vacation_names != task_count_names:
        extra_in_vacation = vacation_names - task_count_names
        extra_in_tasks = task_count_names - vacation_names
        msg = "Names do not match between vacation schedule and task counts."
        if extra_in_vacation:
            msg += f" In vacation but not task counts: {sorted(extra_in_vacation)}."
        if extra_in_tasks:
            msg += f" In task counts but not vacation: {sorted(extra_in_tasks)}."
        sys.exit(msg)

    # Make sure values in vacation schedule are {no, noa, noab, empty}
    valid_vacation_values = {"no", "noa", "noab"}
    for col in vacation.columns[1:]:
        for idx, val in vacation[col].items():
            if pd.isna(val) or val == "":
                continue
            if str(val).strip().lower() not in valid_vacation_values:
                sys.exit(
                    f'Vacation file: Invalid value "{val}" in column "{col}", row {idx + 2}'
                )

    # Make sure values in fellow schedule are {no, empty}
    for idx, val in fellow_schedule["Fellow"].items():
        if pd.isna(val) or val == "":
            continue
        if str(val).strip().lower() != "no":
            sys.exit(
                f'Fellow schedule: Invalid value "{val}" in row {idx + 2}'
            )

    # Make sure values in task counts are {positive number, empty}
    for col in task_counts.columns[1:]:
        for idx, val in task_counts[col].items():
            if pd.isna(val) or val == "":
                continue
            try:
                num = float(val)
                if num <= 0:
                    sys.exit(
                        f'Task counts: Value "{val}" in column "{col}" must be positive'
                    )
            except (ValueError, TypeError):
                sys.exit(
                    f'Task counts: Invalid value "{val}" in column "{col}", expected a positive number'
                )

    return Schedule(
        vacation=vacation, 
        fellow_schedule=fellow_schedule,
        task_counts=task_counts
    )


def parse_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a call schedule")
    parser.add_argument(
        "output_file",
        help="File to write call schedule to (must be .csv)",
        type=Path,
    )
    parser.add_argument(
        "--vacation",
        help="File with vacation requests (must be .csv)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--fellow-schedule",
        help="File with fellow days off (must be .csv)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--task-counts",
        help="File with the number of tasks to be assigned per person (must be .csv)",
        type=Path,
        required=True,
    )
    
    args = parser.parse_args()

    # Validate that files exist
    if not args.vacation.exists():
        sys.exit(f"Vacation file {args.vacation} does not exist")
    if not args.fellow_schedule.exists():
        sys.exit(f"Fellow schedule file {args.fellow_schedule} does not exist")
    if not args.task_counts.exists():
        sys.exit(f"Task count file {args.task_counts} does not exist")
    
    # Validate that files are .csv
    if args.vacation.suffix.lower() != ".csv":
        sys.exit(f"Vacation file {args.vacation} is not a csv file")
    if args.fellow_schedule.suffix.lower() != ".csv":
        sys.exit(
            f"Fellow schedule file {args.fellow_schedule} is not a csv file"
        )
    if args.task_counts.suffix.lower() != ".csv":
        sys.exit(f"Task count file {args.task_counts} is not a csv file")
    
    return args