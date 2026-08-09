"""
conftest.py — pytest session configuration

Suppresses known third-party deprecation warnings so test output stays clean.
These are all upstream library issues, not problems in our code:

  - pandas_ta uses deprecated Copy-on-Write option (pandas >= 3.0 ignores it)
  - pandas_ta uses bitwise ~ on numpy booleans (deprecated in pandas >= 2.x)
"""

import warnings
import pytest


def pytest_configure(config):
    # pandas_ta: mode.copy_on_write deprecation
    warnings.filterwarnings(
        "ignore",
        message=".*mode.copy_on_write.*",
        category=DeprecationWarning,
    )
    # pandas_ta: bitwise inversion on bool inside pandas internals
    warnings.filterwarnings(
        "ignore",
        message=".*Bitwise inversion.*on bool.*",
        category=DeprecationWarning,
        module=r"pandas.*",
    )
    # pandas_ta: Pandas4Warning about copy_on_write
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"pandas_ta.*",
    )
