from __future__ import print_function

import hydra
from omegaconf import DictConfig
import os
import random
import logging
import torch.utils.data

import src.datasets
import src.optimal_transport_modules

from matplotlib import pyplot as plt
from scipy.stats import truncnorm

from src.optimal_transport_modules.icnn_modules import *
from src.utils import set_random_seeds, setup_logging
from src.utils import ResultsLog, get_iccns, get_optimizers


# useful functions

def compute_constraint_loss(list_of_params):

    loss_val = 0

    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values

@hydra.main(version_base=None, config_path="config", config_name="celeba_train_config")
def main(cfg: DictConfig):

    cuda = not cfg.settings.no_cuda and torch.cuda.is_available()

    # random seeds
    set_random_seeds(cfg.settings.seed)

    dataset = cfg.data.dataset_x.split("/")[2]

    # get hydra config
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()

    # results save paths
    results_save_path = hydra_cfg.run.dir
    model_save_path = os.path.join(results_save_path, "storing_models")

    os.makedirs(model_save_path, exist_ok=True)

    if cfg.settings.verbose:
        logging.info("saving to %s \n", results_save_path)
        logging.debug("run arguments: %s", cfg)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    ################################################################
    # Data stuff

    if dataset == "celeba":
        X_data = src.datasets.CelebA_Features(
                        cfg.data.dataset_x,
                        "../data/{}/{}".format(dataset,
                                               cfg.data.features))

        train_loader = torch.utils.data.DataLoader(
                                X_data,
                                batch_size=cfg.training.batch_size,
                                shuffle=True,
                                **kwargs)
        if cfg.settings.verbose:
            logging.info("Created the data loader for X\n")


        Y_data = src.datasets.CelebA_Features_Kernel(
                        cfg.data.dataset_y,
                        "../data/{}/{}".format(
                                            dataset,
                                            cfg.data.features),
                        var=cfg.data.kernel_variance)
    if dataset == "celeba":
        X_data = src.datasets.CelebA_Features(
                        cfg.data.dataset_x,
                        "../data/{}/{}".format(dataset,
                                               cfg.data.features))

        train_loader = torch.utils.data.DataLoader(
                                X_data,
                                batch_size=cfg.training.batch_size,
                                shuffle=True,
                                **kwargs)
        if cfg.settings.verbose:
            logging.info("Created the data loader for X\n")


        Y_data = src.datasets.CelebA_Features_Kernel(
                        cfg.data.dataset_y,
                        "../data/{}/{}".format(
                                            dataset,
                                            cfg.data.features),
                        var=cfg.data.kernel_variance)

    else:
        raise NotImplementedError("Not implemented for this dataset")
    ################################################################
    # Everything related to both the convex functions

    convex_f, convex_g = get_iccns(
                              cfg.iccn.num_layers,
                              cfg.iccn.full_quadratic,
                              cfg.iccn.input_dim,
                              cfg.iccn.num_neuron,
                              cfg.iccn.activation)

    # Form a list of positive weight parameters
    # and also initialize them with positive values
    f_positive_params = []

    for p in list(convex_f.parameters()):
        if hasattr(p, 'be_positive'):
            f_positive_params.append(p)

        p.data = torch.from_numpy(
            truncated_normal(p.shape,
                             threshold=1./np.sqrt(p.shape[1]
                                                  if len(p.shape) > 1
                                                  else p.shape[0]))).float()

    g_positive_params = []

    for p in list(convex_g.parameters()):
        if hasattr(p, 'be_positive'):
            g_positive_params.append(p)

        p.data = torch.from_numpy(
            truncated_normal(p.shape,
                             threshold=1./np.sqrt(p.shape[1]
                                                  if len(p.shape) > 1
                                                  else p.shape[0]))).float()

    if cuda:
        convex_f = convex_f.cuda()
        convex_g = convex_g.cuda()

    if cfg.settings.verbose:
        logging.info("Created and initialized the convex neural networks 'f' and 'g'")

    num_parameters = sum([parameter.nelement()
                          for parameter in convex_f.parameters()])
    if cfg.settings.verbose:
        logging.info("number of parameters: %d", num_parameters)

    f_positive_constraint_loss = compute_constraint_loss(f_positive_params)

    g_positive_constraint_loss = compute_constraint_loss(g_positive_params)


    optimizer_f, optimizer_g = get_optimizers(convex_f,
                                              convex_g,
                                              cfg.training.optimizer,
                                              cfg.training.lr,
                                              cfg.training.momentum,
                                              cfg.training.beta1_adam,
                                              cfg.training.beta2_adam,
                                              cfg.training.alpha_rmsprop)

    # Training stuff
    def train(epoch):

        convex_f.train()
        convex_g.train()

        w_2_loss_value_epoch = 0
        g_OT_loss_value_epoch = 0
        g_Constraint_loss_value_epoch = 0

        for batch_idx, (real_data, _, _, _) in enumerate(train_loader):

            if cuda:
                real_data = real_data.cuda()

            real_data = Variable(real_data)

            indices = random.sample(range(len(Y_data)), len(real_data))
            Y_subset = torch.utils.data.Subset(Y_data, indices)
            Y_loader = torch.utils.data.DataLoader(Y_subset,
                                                   batch_size=len(real_data),
                                                   shuffle=True,
                                                   **kwargs)
            y, _, _, _, _ = next(iter(Y_loader))

            if cuda:
                y = y.cuda()

            y = Variable(y, requires_grad=True)

            optimizer_f.zero_grad()
            optimizer_g.zero_grad()

            g_OT_loss_val_batch = 0
            g_Constraint_loss_val_batch = 0

            for inner_iter in range(1, cfg.training.n_generator_iters+1):
                # First do a forward pass on y and compute grad_g_y
                # Then do a backward pass update on parameters of g

                optimizer_g.zero_grad()

                g_of_y = convex_g(y).sum()

                grad_g_of_y = torch.autograd.grad(g_of_y, y, create_graph=True)[0]

                f_grad_g_y = convex_f(grad_g_of_y).mean()
                dot_prod = (grad_g_of_y * y).sum(dim=1).mean()

                loss_g = f_grad_g_y - dot_prod
                g_OT_loss_val_batch += loss_g.item()

                if cfg.training.lambda_mean > 0:

                    mean_difference_loss = cfg.training.lambda_mean *\
                        (real_data.mean(0) - grad_g_of_y.mean(0)).pow(2).sum()
                    variance_difference_loss = cfg.training.lambda_mean *\
                        (real_data.std(0) - grad_g_of_y.std(0)).pow(2).sum()

                    loss_g += mean_difference_loss + variance_difference_loss

                loss_g.backward()

                # Constraint loss for g parameters
                if cfg.training.lambda_cvx> 0:
                    g_positive_constraint_loss = cfg.training.lambda_cvx *\
                        compute_constraint_loss(g_positive_params)
                    g_Constraint_loss_val_batch += g_positive_constraint_loss
                    g_positive_constraint_loss.backward()

                optimizer_g.step()

                # Maintaining the positive constraints on the convex_g_params
                if cfg.training.lambda_cvx == 0:
                    for p in g_positive_params:
                        p.data.copy_(torch.relu(p.data))

                # Just for the last iteration keep the gradient on f intact
                # otherwise need to do from scratch
                if inner_iter != cfg.training.n_generator_iters:
                    optimizer_f.zero_grad()

            g_OT_loss_val_batch /= cfg.training.n_generator_iters
            g_Constraint_loss_val_batch /= cfg.training.n_generator_iters

            # Flip the gradient sign for parameters in convex_f
            # because we need to solve "sup" maximization for f
            for p in list(convex_f.parameters()):
                p.grad.copy_(-p.grad)

            remaining_f_loss = convex_f(real_data).mean()
            remaining_f_loss.backward()

            optimizer_f.step()

            # Maintain the "f" parameters positive
            for p in f_positive_params:
                p.data.copy_(torch.relu(p.data))

            w_2_loss_value_batch = (g_OT_loss_val_batch -
                                    remaining_f_loss.item() +
                                    0.5*real_data.pow(2).sum(dim=1).mean().item() +
                                    0.5*y.pow(2).sum(dim=1).mean().item())
            w_2_loss_value_epoch += w_2_loss_value_batch

            g_OT_loss_value_epoch += g_OT_loss_val_batch
            g_Constraint_loss_value_epoch += g_Constraint_loss_val_batch

            if cfg.settings.verbose and batch_idx % cfg.settings.log_interval == 0:
                logging.info(('Train Epoch: {}'
                              ' [{}/{} ({:.0f}%)]'
                              ' g_OT_loss: {:.4f}'
                              ' g_Constraint_Loss: {:.4f}'
                              ' W_2_Loss: {:.4f} ').format(
                                            epoch,
                                            batch_idx * len(real_data),
                                            len(train_loader.dataset),
                                            100. * batch_idx / len(train_loader),
                                            g_OT_loss_val_batch,
                                            g_Constraint_loss_val_batch,
                                            w_2_loss_value_batch))

        w_2_loss_value_epoch /= len(train_loader)
        g_OT_loss_value_epoch /= len(train_loader)
        g_Constraint_loss_value_epoch /= len(train_loader)

        return (w_2_loss_value_epoch,
                g_OT_loss_value_epoch,
                g_Constraint_loss_value_epoch)

    ###################################################

    total_w_2_epoch_loss_list = []
    total_g_OT_epoch_loss_list = []
    total_g_Constraint_epoch_loss_list = []

    for epoch in range(1, cfg.training.epochs + 1):

        (w_2_loss_value_epoch,
         g_OT_loss_value_epoch,
         g_Constraint_loss_value_epoch) = train(epoch)

        total_w_2_epoch_loss_list.append(w_2_loss_value_epoch)
        total_g_OT_epoch_loss_list.append(g_OT_loss_value_epoch)
        total_g_Constraint_epoch_loss_list.append(g_Constraint_loss_value_epoch)

        if epoch % cfg.training.lr_schedule == 0:

            optimizer_g.param_groups[0]['lr'] *= 0.5

            optimizer_f.param_groups[0]['lr'] *= 0.5

        # if epoch % 10 == 0:
        if epoch % 1 == 0:
            torch.save(convex_f.state_dict(),
                       model_save_path + '/convex_f_epoch_{0}.pt'.format(epoch))
            torch.save(convex_g.state_dict(),
                       model_save_path + '/convex_g_epoch_{0}.pt'.format(epoch))

        else:
            torch.save(convex_f.state_dict(),
                       model_save_path + '/convex_f_lastepoch.pt')
            torch.save(convex_g.state_dict(),
                       model_save_path + '/convex_g_lastepoch.pt')

    plt.plot(range(1, len(total_w_2_epoch_loss_list) + 1),
             total_w_2_epoch_loss_list,
             label='Training loss')

    plt.xlabel('iterations')
    plt.ylabel(r'$W_2$-loss value')
    plt.savefig(results_save_path + '/training_loss.png')
    plt.clf()

    if cfg.settings.verbose:
        logging.info("Training is finished and the models"
                    " and plots are saved. Good job :)")

if __name__ == "__main__":
    main()
