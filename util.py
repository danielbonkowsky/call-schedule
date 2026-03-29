import typing
import sys
import pandas as pd
from datetime import datetime, date
from pathlib import Path
import numpy as np


class Schedule:
    """Schedule class with validated dataframe for easy access"""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    @property
    def names(self) -> list[str]:
        """Returns the names of all physicians in the schedule"""

        return list(self._df.columns)[1:-1]
    
    @property
    def weeks(self) -> list[str]:
        """Returns a list of all the weeks in the schedule"""

        return self._df["Week"].to_list()
    
    def week_has_fellow(self, week: str) -> bool:
        """Determine whether a given week has a fellow assigned"""

        result = self._df.loc[self._df["Week"] == week, "Fellow"]
        return result.values[0] in {"", np.nan, None, ""}
        

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

    # Validate data
    allowed_vals = {"no", "noa", "noab", "nob", np.nan, None, ""}
    filtered_schedule = schedule.drop(columns=["Week"])
    valid_mask = filtered_schedule.isin(allowed_vals)
    all_valid = valid_mask.all().all()

    if not all_valid:
        invalid_mask = ~valid_mask
        row_indices, col_indices = np.where(invalid_mask)
        for r, c in zip(row_indices, col_indices):
            data = filtered_schedule.iloc[r, c]
            
            print(f"Invalid data at col {c}, row {r}: {data}")
        sys.exit(1)
    
    return Schedule(schedule)