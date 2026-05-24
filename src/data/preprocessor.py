from typing import Dict

import pandas as pd


class DataPreprocessor:
    """Cleans and merges the relational dataset for analysis."""

    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    def clean_all(self) -> Dict[str, pd.DataFrame]:
        for name, df in self.data.items():
            self.data[name] = self._clean(df)
        return self.data

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.select_dtypes(include="object"):
            df[col] = df[col].str.strip() if df[col].notna().any() else df[col]
        return df

    def build_customer_360(self) -> pd.DataFrame:
        customers = self.data.get("Customers", pd.DataFrame())
        accounts = self.data.get("Accounts", pd.DataFrame())
        products = self.data.get("Products", pd.DataFrame())
        installations = self.data.get("Installations", pd.DataFrame())
        sales = self.data.get("Sales", pd.DataFrame())
        leads = self.data.get("Leads", pd.DataFrame())

        merged = customers.merge(
            accounts, left_on="id", right_on="customer_id", how="left", suffixes=("", "_acc")
        )

        merged = merged.merge(
            products, left_on="proudct_id", right_on="id", how="left", suffixes=("", "_prod")
        )

        merged = merged.merge(
            installations, left_on="installation_id", right_on="id", how="left", suffixes=("", "_inst")
        )

        if not sales.empty:
            sales_agg = sales.groupby("account_id").agg(
                total_sales=("id", "count"),
                first_sale_date=("sale_date", "min"),
                last_sale_date=("sale_date", "max"),
            ).reset_index()
            merged = merged.merge(
                sales_agg, left_on="id_acc", right_on="account_id", how="left", suffixes=("", "_sale")
            )

        if not leads.empty:
            leads_info = leads[["id", "source", "customer_id"]].rename(
                columns={"id": "lead_id_ref", "source": "lead_source"}
            )
            merged = merged.merge(
                leads_info, left_on="lead_id", right_on="lead_id_ref", how="left", suffixes=("", "_lead")
            )

        return merged
