import pytest
import torch

from hydra import initialize, compose
from src.evaluation import load_iccns, load_data
@pytest.fixture
def config(request):
    return request.config.getoption('--config')

@pytest.fixture
def epoch_to_test(request):
    return request.config.getoption('--epoch_to_test')

def test_convergence(config, epoch_to_test):
    with initialize(version_base=None, config_path="../scripts/outputs/{}/.hydra".format(config)):
        cfg = compose(config_name="config")

        assert True


def test_W2_is_positive(config, epoch_to_test):
    with initialize(version_base=None, config_path="../scripts/outputs/{}/.hydra".format(config)):
        cfg = compose(config_name="config")

        convex_f, convex_g = load_iccns(cfg, epoch_to_test)

        assert True


def test_different_computations_of_W2_are_equal(config, epoch_to_test):
    with initialize(version_base=None, config_path="../scripts/outputs/{}/.hydra".format(config)):
        cfg = compose(config_name="config")

        convex_f, convex_g = load_iccns(cfg, epoch_to_test)

        assert True


def test_f_and_g_are_convex_conjugates(config, epoch_to_test):
    with initialize(version_base=None, config_path="../scripts/outputs/{}/.hydra".format(config)):
        cfg = compose(config_name="config")

        convex_f, convex_g = load_iccns(cfg, epoch_to_test)

        assert True