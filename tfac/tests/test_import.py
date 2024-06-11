"""
Test that we can successfully import the datasets.
"""
import pytest
import numpy as np
import pandas as pd
from tfac.dataImport import import_patient_metadata, form_tensor, import_rna


@pytest.mark.parametrize("call", [import_patient_metadata, import_rna])
def test_importBases(call):
    """ Test that the most basic imports work. """
    data = call()

    assert isinstance(data, pd.DataFrame)


def test_fullImport():
    """ Test that we can import the full dataset. """
    tensor, matrix, patient_data = form_tensor()

    assert isinstance(tensor, np.ndarray)
    assert isinstance(matrix, np.ndarray)
    assert tensor.shape[0] == matrix.shape[0]
    assert isinstance(patient_data, pd.DataFrame)
