"""
Utilities for metric-related helpers.
"""
from typing import Optional
import numpy as np


def select_positive_proba(model, proba: np.ndarray, positive_label: int = 1, logger=None) -> np.ndarray:
    """Return the probability vector for the positive label."""
    if proba is None:
        return proba
    proba_arr = np.asarray(proba)
    if proba_arr.ndim == 1:
        return proba_arr
    if proba_arr.shape[1] == 1:
        return proba_arr[:, 0]

    classes = None
    if hasattr(model, "model") and hasattr(model.model, "classes_"):
        classes = model.model.classes_
    elif hasattr(model, "classes_"):
        classes = model.classes_

    if classes is not None:
        try:
            classes_list = list(classes)
            idx = classes_list.index(positive_label)
            return proba_arr[:, idx]
        except Exception as e:
            if logger:
                logger.warning(
                    "Positive label %s not found in classes %s; using default column. error=%s",
                    positive_label,
                    classes,
                    e
                )
    return proba_arr[:, 1]
