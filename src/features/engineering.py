import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except Exception:
    pass


class FeatureEngineer:
    """Creates derived features for segmentation and credit-risk modelling."""

    PRODUCT_CAT_PATTERN = (
        r"^(Solar|Clean|Water|Premium|Battery|PAYG|Tablet|Smartphone|Solar TV)"
    )

    def __init__(self, customer_360: pd.DataFrame):
        self.df = customer_360.copy()

    def engineer(self) -> pd.DataFrame:
        """Full feature set including behavioural indicators (segmentation)."""
        self._payment_behaviour()
        self._account_tenure()
        self._product_features()
        self._geographic_indicators()
        return self.df

    def engineer_features(self) -> pd.DataFrame:
        """Alias for engineer()."""
        return self.engineer()

    def _payment_behaviour(self):
        if "status" in self.df:
            status_map = {
                "Arrears": 2,
                "Repossession": 2,
                "Repossed": 2,
                "Write Off": 3,
                "Refunded": 1,
                "No Deposit": 1,
                "Advance": 0,
                "Complete": 0,
            }
            self.df["risk_score"] = self.df["status"].map(status_map).fillna(1)
            self.df["is_default"] = self.df["status"].isin(
                ["Write Off", "Repossession", "Repossed"]
            ).astype(int)
            self.df["is_complete"] = (self.df["status"] == "Complete").astype(int)
            self.df["in_arrears"] = (self.df["status"] == "Arrears").astype(int)
        if "type" in self.df:
            self.df["is_payg"] = (self.df["type"] == "PAYG").astype(int)

    def _account_tenure(self):
        date_col = "first_installment_date" if "first_installment_date" in self.df else "created_at_acc"
        date_col = date_col if date_col in self.df else "created_at"
        if date_col in self.df:
            parsed = pd.to_datetime(self.df[date_col], errors="coerce")
            self.df["account_tenure_days"] = (pd.Timestamp.now() - parsed).dt.days.fillna(0)

    def _product_features(self):
        if "is_refurb" in self.df:
            self.df["is_refurbished"] = self.df["is_refurb"].fillna(False).astype(int)
        if "product" in self.df:
            self.df["product_category"] = self.df["product"].str.extract(
                self.PRODUCT_CAT_PATTERN
            )[0].fillna("Other")

    def _geographic_indicators(self):
        if "region" in self.df:
            self.df = pd.get_dummies(
                self.df, columns=["region"], prefix="region", drop_first=False
            )

    def engineer_credit(self) -> pd.DataFrame:
        """Features for credit-risk — NO status-derived columns (prevents target leakage).

        Only uses data available at account opening or that does not come from
        the ``status`` field.  ``is_default`` is included as a label column.
        """
        df = self.df.copy()

        if "type" in df:
            df["is_payg"] = (df["type"] == "PAYG").astype(int)

        date_col = (
            "first_installment_date" if "first_installment_date" in df
            else "created_at_acc" if "created_at_acc" in df
            else "created_at" if "created_at" in df
            else None
        )
        if date_col and date_col in df:
            parsed = pd.to_datetime(df[date_col], errors="coerce")
            df["account_tenure_days"] = (pd.Timestamp.now() - parsed).dt.days.fillna(0)
        else:
            df["account_tenure_days"] = 0

        if "is_refurb" in df:
            df["is_refurbished"] = df["is_refurb"].fillna(False).astype(int)
        if "product" in df:
            df["product_category"] = df["product"].str.extract(
                self.PRODUCT_CAT_PATTERN
            )[0].fillna("Other")

        if "region" in df:
            df = pd.get_dummies(df, columns=["region"], prefix="region", drop_first=False)

        if "lead_source" in df:
            df = pd.get_dummies(df, columns=["lead_source"], prefix="lead", drop_first=False)

        if "gender" in df:
            df["is_male"] = (df["gender"].str.lower() == "male").astype(int)

        if "date_of_birth" in df and "created_at" in df:
            dob = pd.to_datetime(df["date_of_birth"], errors="coerce")
            created = pd.to_datetime(df["created_at"], errors="coerce")
            df["age_at_signup"] = ((created - dob).dt.days / 365.25).fillna(0).astype(int)

        if "product_category" in df:
            df = pd.get_dummies(df, columns=["product_category"], prefix="prod", drop_first=False)

        for col in ["latitude", "longitude"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # 48% of sales predate account creation — drop to avoid look-ahead bias
        if "total_sales" in df.columns:
            df = df.drop(columns=["total_sales"])

        if "status" in df:
            df["is_default"] = df["status"].isin(
                ["Write Off", "Repossession", "Repossed"]
            ).astype(int)
        else:
            df["is_default"] = 0

        obj_cols = df.select_dtypes(include="object").columns.tolist()
        if obj_cols:
            df = df.drop(columns=obj_cols)

        bool_cols = df.select_dtypes(include="bool").columns.tolist()
        for c in bool_cols:
            df[c] = df[c].astype(int)

        return df
