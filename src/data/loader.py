from pathlib import Path
from typing import Dict

import pandas as pd


class DataLoader:
    """Loads all sheets from the multi-sheet assessment Excel file."""

    SHEET_NAMES = [
        "Customers",
        "Departments",
        "Users",
        "Products",
        "Leads",
        "Installations",
        "Accounts",
        "Sales",
    ]

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")

    def load_all(self) -> Dict[str, pd.DataFrame]:
        return {sheet: self.load_sheet(sheet) for sheet in self.SHEET_NAMES}

    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(self.filepath, sheet_name=sheet_name)
        df.columns = [c.strip().replace(" ", "_").lower() for c in df.columns]
        return df
