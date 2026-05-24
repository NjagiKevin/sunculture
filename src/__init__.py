from .data import DataLoader, DataPreprocessor
from .features import FeatureEngineer
from .models import CustomerSegmentation, CreditRiskModel


def __getattr__(name):
    if name == "Plotter":
        from .viz import Plotter
        return Plotter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DataLoader", "DataPreprocessor",
    "FeatureEngineer",
    "CustomerSegmentation", "CreditRiskModel",
    "Plotter",
]