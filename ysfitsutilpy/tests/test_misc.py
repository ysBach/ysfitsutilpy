from pathlib import Path
import pytest

from ysfitsutilpy import misc

@pytest.mark.parametrize("inputs, path_to_text, result", [
    ("a_sample_fits.fits", True, "a_sample_fits.fits"),
    ("a_sample_fits.fits", False, Path("a_sample_fits.fits")),
])
def test_inputs2list_singleinput(inputs, path_to_text, result):
    assert misc.inputs2list(inputs=inputs, path_to_test=path_to_text) == result


def test_inputs2list_strpath():
    res = misc.inputs2list("a_sample_fits.fits")
    assert res == ["a_sample_fits.fits"]


def test_inputs2list_path():
    samplefits = Path("a_sample_fits.fits")
    res = misc.inputs2list(samplefits)
    assert res == samplefits
