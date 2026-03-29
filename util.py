import typing
import sys
import pandas as pd
from datetime import datetime, date
from pathlib import Path


def _validate_7_day_intervals(date_strings: list[str]) -> tuple[bool, str]:
    """Validate that the dates are all 7 days apart"""

    if not date_strings:
        return False, "The dates are empty."

    current_year = date.today().year
    parsed_dates = []
    
    for i, item in enumerate(date_strings):
        # 2. Check if the item is actually a string
        if not isinstance(item, str):
            return False, f"Invalid data type at index {i}: '{item}' is not a string."
            
        try:
            # 3. Try to parse the date format (DD-Mon)
            dt = datetime.strptime(item, "%d-%b")
            
            # Logic to handle year rollover (e.g., Dec -> Jan)
            if i > 0 and dt.month < parsed_dates[-1].month:
                current_year += 1
                
            dt = dt.replace(year=current_year)
            parsed_dates.append(dt)
            
        except ValueError:
            # 4. Handle incorrect formats or non-date text
            return False, f"Format error at index {i}: '{item}' does not match 'Day-Month' (e.g., 6-Jul)."

    # 5. Validate the 7-day gap
    for i in range(1, len(parsed_dates)):
        diff = (parsed_dates[i] - parsed_dates[i-1]).days
        if diff != 7:
            return False, f"Gap error: {date_strings[i-1]} to {date_strings[i]} is {diff} days."
            
    return True, "Success: All dates are valid and exactly 7 days apart."


def validate_schedule(schedule_file: Path) -> pd.DataFrame:
    """Takes in a file path and ensures that it is valid schedule"""

    # Validate schedule file
    if not schedule_file.exists():
        sys.exit(f"File {schedule_file} does not exist")
    if schedule_file.suffix.lower() != ".csv":
        sys.exit(f"File {schedule_file} is not a csv file")

    schedule = pd.read_csv(schedule_file)

    # Validate schedule format
    col_names = list(schedule.columns)
    if col_names[0] != "Week":
        sys.exit(f'Expected first column "Week", got "{col_names[0]}"')
    if col_names[-1] != "Fellow":
        sys.exit(f'Expected last column "Fellow", got "{col_names[-1]}"')
    
    # Validate weeks
    valid, msg = _validate_7_day_intervals(schedule["Week"].to_list())
    if not valid:
        sys.exit(msg)
    
    return schedule

def get_names(schedule: pd.DataFrame) -> list[str]:
    """Returns the names of all physicians in the schedule"""

    return list(schedule.columns)[1:-1]

def get_weeks(schedule: pd.DataFrame) -> list[str]:
    """Returns a list of all the weeks in the schedule"""

    return schedule["Week"].to_list()