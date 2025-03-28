import torch
from torch.autograd import grad
import numpy as np
from math import pi
import random

def d(f, x):
    return grad(f, x, grad_outputs=torch.ones_like(f), create_graph=True, only_inputs=True)[0]

def Power(x, y, loc_x, loc_y, phi):
    return phi * (torch.sign(x - loc_x) + 1) * (torch.sign(-x + loc_x + 1) + 1) * \
           (torch.sign((y - loc_y)) + 1) * (torch.sign(-y + loc_y + 1) + 1) / (16)


def PDE1(u, x, y, positions, units, phi):
    out = 0.
    for i in range(len(phi)):
        out += phi[i] * (torch.tanh(1e5 * (x - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (torch.tanh(1e5 * (-x + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (torch.tanh(1e5 * (y - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (torch.tanh(1e5 * (-y + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)
    return 1.5*(d(d(u, x), x) + d(d(u, y), y) )+ out, out


def PDE2(u, x, y, positions, units, phi):
    out = 0.
    for i in range(len(phi)):
        out += phi[i] * (torch.tanh(1e5 * (x - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (torch.tanh(1e5 * (-x + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (torch.tanh(1e5 * (y - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (torch.tanh(1e5 * (-y + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)
    return 0.003*(d(d(u, x), x) + d(d(u, y), y)) +out, out



def PDE(u, x, y, positions, units, phi):
    out = 0.
    k = torch.full_like(x, 0.003)  # 默认导热系数为0.3

    for i in range(len(phi)):
        mask = (x >= (positions[i, 0] - units[i, 0] / 2)) & (x <= (positions[i, 0] + units[i, 0] / 2)) & \
               (y >= (positions[i, 1] - units[i, 1] / 2)) & (y <= (positions[i, 1] + units[i, 1] / 2))
        k[mask] = 1.5  # 在热源区域内，导热系数为1.5

        out += phi[i] * (torch.tanh(1e5 * (x - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (torch.tanh(1e5 * (-x + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (torch.tanh(1e5 * (y - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (torch.tanh(1e5 * (-y + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)

    return k * (d(d(u, x), x) + d(d(u, y), y)) + out, out


def PDE_inverse(u, x, y, positions, units, phi):
    out = 0.
    for i in range(len(phi)):
        out += phi[i] * (torch.tanh(1e5 * (x - positions[i, 0] + units[i, 0] / 2)) + 1) \
               * (torch.tanh(1e5 * (-x + positions[i, 0] + units[i, 0] / 2)) + 1) * \
               (torch.tanh(1e5 * (y - positions[i, 1] + units[i, 1] / 2)) + 1) \
               * (torch.tanh(1e5 * (-y + positions[i, 1] + units[i, 1] / 2)) + 1) / (16)
    return d(d(u, x), x) + d(d(u, y), y) + out, out


def is_neumann_boundary_x(u, x, y):
    return d(u, x)


def is_neumann_boundary_y(u, x, y):
    return d(u, y)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


