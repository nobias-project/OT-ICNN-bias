import pytest

def pytest_addoption(parser):
    parser.addoption(
        "--config",
        action = "store",
        default="",
    )

    parser.addoption(
        "--epoch_to_test",
        action="store",
        default="",
    )
