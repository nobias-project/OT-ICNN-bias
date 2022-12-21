import pytest
import torch

from hydra import initialize, compose
from src.evaluation import load_iccns, load_data
from src.evaluation import compute_w2_Monge, compute_w2_Kantorovich, compute_OT_loss, compute_convex_conjugate_loss

# pytest fixtures
@pytest.fixture(scope="session")
def config(request):
    return request.config.getoption('--config')

@pytest.fixture(scope="session")
def dataset(request):
    return request.config.getoption('--dataset')

@pytest.fixture(scope="session")
def epoch_to_test(request):
    return request.config.getoption('--epoch_to_test')

@pytest.fixture(scope="session")
def convex_f_and_convex_g(dataset, config, epoch_to_test):
    with initialize(version_base=None, config_path="../results/training/{}/{}/.hydra".format(dataset, config)):
        cfg = compose(config_name="config")
        results_save_path ="../results/training/{}/{}".format(dataset, config)
        convex_f, convex_g = load_iccns(results_save_path, cfg, epoch_to_test)
        return [convex_f, convex_g]

@pytest.fixture(scope="session")
def data_x_and_data_y(dataset, config):
    with initialize(version_base=None, config_path="../results/training/{}/{}/.hydra".format(dataset, config)):
        cfg = compose(config_name="config")
        X_data, Y_data = load_data(cfg)
        return [X_data, Y_data]

@pytest.fixture(scope="session")
def cuda(dataset, config):
    with initialize(version_base=None, config_path="../results/training/{}/{}/.hydra".format(dataset, config)):
        cfg = compose(config_name="config")
        cuda = not cfg.settings.no_cuda and torch.cuda.is_available()

        return cuda

@pytest.fixture(scope="session")
def compute_OT_loss_fixture(convex_f_and_convex_g, data_x_and_data_y, cuda):
    convex_f, convex_g = convex_f_and_convex_g
    X_data, Y_data = data_x_and_data_y

    w2 = compute_OT_loss(X_data, Y_data, convex_f, convex_g, cuda)
    return w2


@pytest.fixture(scope="session")
def compute_w2_Kantorovich_fixture(convex_f_and_convex_g, data_x_and_data_y, cuda):
    convex_f, convex_g = convex_f_and_convex_g
    X_data, Y_data = data_x_and_data_y

    w2_kantorovich = compute_w2_Kantorovich(X_data, Y_data, convex_f, convex_g, cuda)

    return w2_kantorovich


@pytest.fixture(scope="session")
def compute_w2_Monge_fixture(convex_f_and_convex_g, data_x_and_data_y, cuda):
    _, convex_g = convex_f_and_convex_g
    _, Y_data = data_x_and_data_y

    w2_monge = compute_w2_Monge(Y_data, convex_g, cuda)

    return w2_monge


@pytest.fixture(scope="session")
def compute_convex_conjugate_loss_fixture(convex_f_and_convex_g, data_x_and_data_y, cuda):
    convex_f, convex_g = convex_f_and_convex_g
    X_data, Y_data = data_x_and_data_y

    loss = compute_convex_conjugate_loss(Y_data, convex_f, convex_g, cuda)

    return loss


# tests
def test_W2_is_positive(compute_OT_loss_fixture):
    w2 = compute_OT_loss_fixture

    assert w2 >= 0


@pytest.mark.parametrize("eps", [50, 30, 10, 1])
def test_W2_Makkuva_vs_Kantorovich(compute_OT_loss_fixture, compute_w2_Kantorovich_fixture, eps):
    w2_makkuva = compute_OT_loss_fixture
    w2_kantorovich = compute_w2_Kantorovich_fixture

    assert abs(w2_makkuva - w2_kantorovich) <= eps


@pytest.mark.parametrize("eps", [50, 30, 10, 1])
def test_W2_Makkuva_vs_Monge(compute_OT_loss_fixture, compute_w2_Monge_fixture, eps):
    w2_makkuva = compute_OT_loss_fixture
    w2_monge = compute_w2_Monge_fixture

    assert abs(w2_makkuva - w2_monge) <= eps

@pytest.mark.parametrize("eps", [50, 30, 10, 1])
def test_W2_Kantorovich_vs_Monge(compute_w2_Kantorovich_fixture, compute_w2_Monge_fixture, eps):
    w2_kantorovich = compute_w2_Kantorovich_fixture
    w2_monge = compute_w2_Monge_fixture

    assert abs(w2_kantorovich - w2_monge) <= eps

@pytest.mark.parametrize("eps", [15, 10, 1, .1])
def test_f_and_g_are_convex_conjugates(compute_convex_conjugate_loss_fixture, eps):
    loss = compute_convex_conjugate_loss_fixture

    assert loss <= eps