import numpy as np
from scipy import fftpack
import torch
import warnings

# prox_style = 'primal' # prox_{gamma g} = gamma D_sigma + (1-gamma) Id
# prox_style = 'dual' # prox_{gamma g} = D_{gamma sigma}
# prox_style = 'ignore' # prox_{gamma g} = D_{sigma}
prox_style = 'avg' # prox_{gamma g} = (1-alpha) I + alpha D_{sigma}
def TR_gamma(x, gradf, net, gamma, noise_str, return_N = False, alpha=1.):
    '''
    Compute T_gamma(x) = prox_{\gamma g} (x) and R_gamma = gamma^-1 (x - T_gamma)
    where the prox operator is replaced by a denoiser net DRUNet
    Assumes net = prox_{\gamma g} - scale this before passing.
    '''
    # TODO: implement prox
    if gamma <= 0:
        raise Exception("Gamma should be nonnegative")
    if isinstance(gamma, int):
        gamma = float(gamma) #just in case

    # Do this
    
    if prox_style == 'primal':
        torch.set_grad_enabled(True)
        Dg, N = net.calculate_grad(x - gamma * gradf, noise_str)
        torch.set_grad_enabled(False)
        T_gamma = x - gamma * gradf - gamma * Dg
        R_gamma = np.reciprocal(gamma) * (x - T_gamma)
    elif prox_style == 'dual':
        torch.set_grad_enabled(True)
        Dg, N = net.calculate_grad(x - gamma * gradf, noise_str * gamma)
        torch.set_grad_enabled(False)
        T_gamma = x - gamma * gradf - Dg
        R_gamma = np.reciprocal(gamma) * (x - T_gamma)
    elif prox_style == 'ignore':
        torch.set_grad_enabled(True)
        Dg, N = net.calculate_grad(x - gamma * gradf, noise_str)
        # Dg, N = net.calculate_grad(x, noise_str)
        torch.set_grad_enabled(False)
        T_gamma = x - gamma * gradf - Dg
        R_gamma = np.reciprocal(gamma) * (x - T_gamma)
    elif prox_style == 'avg':
        torch.set_grad_enabled(True)
        Dg, N = net.calculate_grad(x - gamma * gradf, noise_str)
        # Dg, N = net.calculate_grad(x, noise_str)
        torch.set_grad_enabled(False)
        T_gamma = x - gamma * gradf - alpha* Dg
        R_gamma = np.reciprocal(gamma) * (x - T_gamma)
    #R_gamma = np.reciprocal(gamma) * (x - T_gamma)

    if return_N:
        return T_gamma, R_gamma, N
    else:
        return T_gamma, R_gamma

def compute_searchdir(y_arr, s_arr, grad_phi_gamma, m):
    '''
    Compute the inverse Hessian approximation H = B^{-1}
    Update via 
    
    y = grad f_{k+1} - grad f_k
    s = x_{k+1} - x_k
    H_k+1  = (I - rho s y^T) H (I - rho y s^T) + rho s s^T

    rho = 1/(y^T s)
    For qN, use 
    s = w - x
    y = grad \phi_\gamma (w) - grad \phi_\gamma (x)
    '''
    # TODO implement L-BFGS

    if len(y_arr) == 0:
        dk = -grad_phi_gamma # PGD iteration
    else:
        dk = -grad_phi_gamma
        # for y_curr, s_curr in zip(y_arr[-m:], s_arr[-m:]):

    return dk

class SearchDirGenerator:
    def __init__(self, initHessFn, m):
        self.initHessFn = initHessFn
        self.y_arr = []
        self.s_arr = []
        self.rho_arr = []
        self.m = m

    def push_ys(self, y, s):

        rho = torch.real(1/torch.tensordot(y, s, dims = 4))
        if rho <= 0:
            warnings.warn("Secant equation fail: discard")
        else:
            self.rho_arr.append(rho)
            self.y_arr.append(y)
            self.s_arr.append(s)
        self.y_arr = self.y_arr[-self.m:]
        self.s_arr = self.s_arr[-self.m:]
        self.rho_arr = self.rho_arr[-self.m:]


    def compute_search(self, grad, override = True):
        if len(self.y_arr) == 0:
            return -grad

        q = grad
        alphas = [] # in reversed chrono order
        for y, s, rho in zip(reversed(self.y_arr), reversed(self.s_arr), reversed(self.rho_arr)):
            alpha = rho * torch.tensordot(s, q, dims = 4)
            q = q - alpha * y
            alphas.append(alpha)
        
        if self.initHessFn is None or override:
            gam = torch.tensordot(self.s_arr[-1], self.y_arr[-1], dims = 4) / torch.tensordot(self.y_arr[-1], self.y_arr[-1], dims = 4)
            r = gam * q
            #print("override")
        else:
            r = self.initHessFn(q)

        for y, s, rho, alpha in zip(self.y_arr, self.s_arr, self.rho_arr, reversed(alphas)):
            beta = rho * torch.tensordot(y, r, dims=4)
            r = r + s * (alpha - beta)

        # sanity check
        sanity_check_pass = torch.tensordot(r, grad, dims=4)>0
        # print("sanity check pass:", sanity_check_pass)
        if sanity_check_pass:
            return -r
        else:
            return -grad

def calculate_phi_gamma(x, gradf, dataterm, net, gamma, noise_str, param_lambda = 1., param_alpha = 1.):
    # Calculate g_gamma
    torch.set_grad_enabled(True)
    Dg, N = net.calculate_grad(x - param_lambda * gamma * gradf, noise_str)
    torch.set_grad_enabled(False)
    g_gamma = param_alpha * torch.norm(x - param_lambda * gamma * gradf - N)**2/2

    phi_gamma = param_lambda * dataterm - param_lambda**2 * gamma * torch.norm(gradf)**2/2 + g_gamma/gamma
    return phi_gamma

def calculate_phi_x(dataterm, w, gradw, Tw, Nw, gamma, param_lambda = 1., param_alpha = 1.):
    # varphi(x) = f(x) + phi_sigma/gamma
    # x = T_gamma(w) = D_sigma(w - gamma gradf(w))
    # phi_sigma = g_sigma(D_sigma^{-1} (x)) - ||D_sigma^{-1}(x) - x||^2/2
    phi_sigma = param_alpha * torch.norm(w - param_lambda * gamma * gradw - Nw)**2/2 - torch.norm(w - param_lambda * gamma * gradw - Tw)**2/2

    return param_lambda * dataterm + phi_sigma/gamma