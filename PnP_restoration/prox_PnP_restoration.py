import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import utils_sr
from utils import utils_qn
import torch
from argparse import ArgumentParser
from utils.utils_restoration import rgb2y, psnr, array2tensor, tensor2array
import sys
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

class PnP_restoration():

    def __init__(self, hparams):

        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fwd_op = None
        self.fwd_op_T = None
        self.initialize_cuda_denoiser()

    def initialize_cuda_denoiser(self):
        '''
        Initialize the denoiser model with the given pretrained ckpt
        '''
        sys.path.append('../GS_denoising/')
        from lightning_denoiser import GradMatch
        parser2 = ArgumentParser(prog='utils_restoration.py')
        parser2 = GradMatch.add_model_specific_args(parser2)
        parser2 = GradMatch.add_optim_specific_args(parser2)
        hparams = parser2.parse_known_args()[0]
        hparams.grad_matching = self.hparams.grad_matching
        hparams.act_mode = 's'
        self.denoiser_model = GradMatch(hparams)
        checkpoint = torch.load(self.hparams.pretrained_checkpoint, map_location=self.device)
        self.denoiser_model.load_state_dict(checkpoint['state_dict'])
        self.denoiser_model.eval()
        for i, v in self.denoiser_model.named_parameters():
            v.requires_grad = False
        self.denoiser_model = self.denoiser_model.to(self.device)
        if self.hparams.precision == 'double' :
            if self.denoiser_model is not None:
                self.denoiser_model.double()


    def initialize_prox(self, img, degradation):
        '''
        calculus for future prox computatations
        :param img: degraded image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        '''
        if self.hparams.degradation_mode == 'deblurring' :
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).double().to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor, 1)
        elif self.hparams.degradation_mode == 'SR':
            self.k = degradation
            self.k_tensor = array2tensor(np.expand_dims(self.k, 2)).double().to(self.device)
            self.FB, self.FBC, self.F2B, self.FBFy = utils_sr.pre_calculate_prox(img, self.k_tensor,self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            self.M = array2tensor(degradation).double().to(self.device)
            self.My = self.M*img
        else:
            print('degradation mode not treated')


    def calculate_prox(self, img):
        '''
        Calculation of the proximal mapping of the data term f
        :param img: input for the prox
        :return: prox_f(img)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            rho = torch.tensor([1/self.hparams.lamb]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.prox_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, 1)
        elif self.hparams.degradation_mode == 'SR':
            rho = torch.tensor([1 /self.hparams.lamb]).double().repeat(1, 1, 1, 1).to(self.device)
            px = utils_sr.prox_solution(img.double(), self.FB, self.FBC, self.F2B, self.FBFy, rho, self.hparams.sf)
        elif self.hparams.degradation_mode == 'inpainting':
            if self.hparams.noise_level_img > 1e-2:
                px = (self.hparams.lamb*self.My + img)/(self.hparams.lamb*self.M+1)
            else :
                px = self.My + (1-self.M)*img
        else:
            print('degradation mode not treated')
        return px

    def calculate_grad(self, img):
        '''
        Calculation of the gradient of the data term f
        :param img: input for the prox
        :return: \nabla_f(img)
        '''
        if self.hparams.degradation_mode == 'deblurring' :
            grad = utils_sr.grad_solution(img.double(), self.FB, self.FBC, self.FBFy, 1)
        if self.hparams.degradation_mode == 'SR' :
            grad = utils_sr.grad_solution(img.double(), self.FB, self.FBC, self.FBFy, self.hparams.sf)
        grad = torch.real(grad)
        return grad

    def calculate_regul(self,y,x,g):
        '''
        Calculation of the regularization (1/tau)*phi_sigma(y)
        :param y: Point where to evaluate
        :param x: D^{-1}(y)
        :param g: Precomputed regularization function value at x
        :return: regul(y)
        '''
        regul = (1 / self.hparams.lamb) * (g - (1 / 2) * torch.norm(x - y, p=2) ** 2)
        return regul

    def calulate_data_term(self,y,img):
        '''
        Calculation of the data term value f(y)
        :param y: Point where to evaluate F
        :param img: Degraded image
        :return: f(y)
        '''
        if self.hparams.degradation_mode == 'deblurring':
            if self.fwd_op is not None:
                deg_y = utils_sr.imfilter(y.double(), fwd_op = self.fwd_op)
            else:
                deg_y = utils_sr.imfilter(y.double(), self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
            if len(img.shape) == 3:
                f = 0.5 * torch.norm(img.permute([2,0,1]) - deg_y, p=2) ** 2
            else:
                f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.degradation_mode == 'SR':
            if self.fwd_op is not None:
                deg_y = utils_sr.imfilter(y.double(), fwd_op = self.fwd_op)
                deg_y = deg_y[..., 0::self.hparams.sf, 0::self.hparams.sf]
            else:
                deg_y = utils_sr.imfilter(y.double(), self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1))
                deg_y = deg_y[..., 0::self.hparams.sf, 0::self.hparams.sf]
            if len(img.shape) == 3:
                f = 0.5 * torch.norm(img.permute([2,0,1]) - deg_y, p=2) ** 2
            else:
                f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        elif self.hparams.degradation_mode == 'inpainting':
            deg_y = self.M * y.double()
            f = 0.5 * torch.norm(img - deg_y, p=2) ** 2
        else:
            print('degradation not implemented')
        return f

    def calculate_Hessian(self, img):
        if self.fwd_op is not None and self.fwd_op_T is not None and self.pad_size is not None:
            if self.hparams.degradation_mode == 'deblurring':
                return utils_sr.Hess(img.double(), fwd_op = self.fwd_op, fwd_op_T = self.fwd_op_T, padding = self.pad_size)
            elif self.hparams.degradation_mode == 'SR':
                return utils_sr.Hess(img.double(), fwd_op = self.fwd_op, fwd_op_T = self.fwd_op_T, padding = self.pad_size, sf = self.hparams.sf)
        else:
            # Create operators
            k = self.k_tensor[0].double().flip(1).flip(2).expand(3, -1, -1, -1)
            c = 3
            pad_size = ((k.shape[-2] - 1) // 2, (k.shape[-1] - 1) // 2)


            foo = torch.nn.Conv2d(c, c, kernel_size = tuple(k.shape[2:]), padding = pad_size, groups = c, padding_mode = 'circular', bias = False)
            foo.weight = torch.nn.Parameter(k).to('cuda')
            
            self.fwd_op = foo
            self.fwd_op.eval()


            bar = torch.nn.ConvTranspose2d(c, c, kernel_size = tuple(k.shape[2:]), padding = 0, groups = c, bias = False)
            bar.weight = torch.nn.Parameter(k).to('cuda')

            self.fwd_op_T = bar
            self.fwd_op_T.eval()
            self.pad_size = pad_size

            if self.hparams.degradation_mode == 'deblurring':
                return utils_sr.Hess(img.double(), fwd_op = self.fwd_op, fwd_op_T = self.fwd_op_T, padding = self.pad_size)
            elif self.hparams.degradation_mode == 'SR':
                return utils_sr.Hess(img.double(), fwd_op = self.fwd_op, fwd_op_T = self.fwd_op_T, padding = self.pad_size, sf = self.hparams.sf)

    def calculate_Dsigma(self, x, strength):
        Dg, N = self.denoiser_model.calculate_grad(x, strength)
        return x - Dg

    def calculate_Lipschitz(self, x):
        # Perform power iteration
        oldfoo = torch.randn_like(x).to('cuda')
        norm = torch.linalg.norm(oldfoo)
        oldfoo = oldfoo / norm

        maxiter = 100
        ctr = 0
        foo = oldfoo.clone()
        while ctr < maxiter and not torch.all(torch.isclose(foo, oldfoo)) or ctr == 0:
            oldfoo = foo
            foo = self.calculate_Hessian(oldfoo)
            foonorm = torch.linalg.norm(foo)
            foo = foo/foonorm
            ctr = ctr + 1
        return foonorm
    # def calculate_varphi_gamma(self, x, img, gamma, strength):
    #     # varphi = f + phi, where D_sigma = prox_phi
    #     # varphi_gamma = f(x) - gamma/2 ||grad f(x)||^2 + g^gamma (x - gamma grad f(x))
    #     #   = 
    #     f = self.calulate_data_term(x,img)
    #     gradf = self.calculate_grad(x)

    #     offset = x - gamma * gradf

    def test_Hessian(self):
        # Check whether the circular depadding is correctly doing A^T A (it is)
        foo1 = torch.randn([1,3,256,256]).double().to('cuda')
        foo2 = torch.randn([1,3,256,256]).double().to('cuda')
        A1 = self.fwd_op(foo1)
        A2 = self.fwd_op(foo2)
        hess = self.calculate_Hessian(foo2)
        print(hess.shape)
        print(torch.tensordot(A1, A2, dims=4))
        print(torch.tensordot(foo1, hess, dims=4))


    def reset_filters(self):
        self.fwd_op = None
        self.fwd_op_T = None
        self.pad_size = None

    def calculate_F(self, y, x, g, img):
        '''
        Calculation of the objective function value f(y) + (1/tau)*phi_sigma(y)
        :param y: Point where to evaluate F
        :param x: D^{-1}(y)
        :param g: Precomputed regularization function value at x
        :param img: Degraded image
        :return: F(y)
        '''
        regul = self.calculate_regul(y,x,g)
        if self.hparams.no_data_term:
            F = regul
        else:
            f = self.calulate_data_term(y.float(),img.float())
            F = f + regul
        return F.item()

    def calculate_lyapunov_DRS(self,y,z,x,g,img):
        '''
            Calculation of the Lyapunov function value Psi(x)
            :param x: Point where to evaluate F
            :param y,z: DRS iterations initialized at x
            :param g: Precomputed regularization function value at x
            :param img: Degraded image
            :return: Psi(x)
        '''
        regul = self.calculate_regul(y,x,g)
        f = self.calulate_data_term(z, img)
        Psi = regul + f + (1 / self.hparams.lamb) * (torch.sum(torch.mul(y-x,y-z)) + (1/2) * torch.norm(y - z, p=2) ** 2)
        return Psi


    def restore(self, img, init_im, clean_img, degradation,extract_results=False):
        '''
        Compute GS-PnP restoration algorithm
        :param img: Degraded image
        :param clean_img: ground-truth clean image
        :param degradation: 2D blur kernel for deblurring and SR, mask for inpainting
        :param extract_results: Extract information for subsequent image or curve saving
        '''

        if extract_results:
            y_list, z_list, x_list, Dg_list, psnr_tab, g_list, Dx_list, F_list, Psi_list = [], [], [], [], [], [], [], [], []
        Psi_list = []
        i = 0 # iteration counter

        img_tensor = array2tensor(init_im).to(self.device) # for GPU computations (if GPU available)
        self.initialize_prox(img_tensor, degradation)  # prox calculus that can be done outside of the loop

        # Initialization of the algorithm
        if self.hparams.degradation_mode == 'SR':
            x0 = cv2.resize(init_im, (img.shape[1] * self.hparams.sf, img.shape[0] * self.hparams.sf),interpolation=cv2.INTER_CUBIC)
            x0 = utils_sr.shift_pixel(x0, self.hparams.sf)
            x0 = array2tensor(x0).to(self.device)
        else:
            x0 = array2tensor(init_im).to(self.device)

        if extract_results:  # extract np images and PSNR values
            out_x = tensor2array(x0.cpu())
            current_x_psnr = psnr(clean_img, out_x)
            if self.hparams.print_each_step:
                print('current x PSNR : ', current_x_psnr)
            psnr_tab.append(current_x_psnr)
            x_list.append(out_x)

        x = x0

        if self.hparams.use_hard_constraint:
            x = torch.clamp(x, 0, 1)

        # Initialize Lyapunov
        diff_Psi = torch.tensor([1.]).to(self.device)
        Psi_old = torch.tensor([1.]).to(self.device)
        Psi = Psi_old
        F = 1.
        # FOR PnP-BFGS
        # gamma = 0.49
        gamma = self.hparams.gamma
        y_arr = []
        s_arr = []
        Beta = self.hparams.beta
        m = 20 #LBFGS
        searchdir_maker = utils_qn.SearchDirGenerator(self.calculate_Hessian,m)
        BFGSBreakFlag = 5
        gammaDecreaseFlag = 5
        
        # print(L_f)
        # for relaxed alphaPGD
        if self.hparams.PnP_algo == 'aPGD':
            # L_f = self.calculate_Lipschitz(x.double())
            L_f = 1
            if self.hparams.degradation_mode == 'SR':
                L_f = 1/(4)
            aPGD_lambda = (gamma+1)/(gamma*L_f)
            # aPGD_alpha = 1 # test if aPGD is working correctly like PGD
            # aPGD_lambda = 1.
            aPGD_alpha = 1/(aPGD_lambda*L_f)
            # aPGD_lambda = 1.
            # aPGD_alpha = 0.9
            print("L_f", L_f)
            print("lambda", aPGD_lambda)
            print("alpha", aPGD_alpha)
            y = x
        if self.hparams.PnP_algo == 'DPIR' or self.hparams.PnP_algo == 'DPIR2':
            sys.path.append('../')
            from DPIR.models.network_unet import UNetRes as net
            from utils.utils_model import test_mode
            from utils.utils_image import augment_img_tensor4
            n_channels=3
            dpir_model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
            dpir_model.load_state_dict(torch.load('../DPIR/model_zoo/drunet_color.pth'), strict=True)
            dpir_model.eval()
            for _, v in dpir_model.named_parameters():
                v.requires_grad = False
            dpir_model = dpir_model.to(self.device)
            dpir_sigmas_ = torch.logspace(np.log10(49/255), np.log10(self.hparams.noise_level_img/255), self.hparams.maxitr)
            dpir_sigmas = dpir_sigmas_.to(self.device)
            dpir_rhos = 0.23*((self.hparams.noise_level_img/255)**2)/(dpir_sigmas**2)

            if self.hparams.degradation_mode == 'SR':
                dpir_sf = self.hparams.sf
            else:
                dpir_sf = 1

            # print(self.k.shape)
            # FB, FBC, F2B, FBFy = self.FB, self.FBC, self.F2B, self.FBFy
            x8 = True
        if self.hparams.PnP_algo == 'DPIR2':
            if self.hparams.degradation_mode == 'SR':
                dpir_sigmas_ = torch.logspace(np.log10(49/255), np.log10(self.hparams.noise_level_img/255), 40)
            else:
                dpir_sigmas_ = torch.logspace(np.log10(49/255), np.log10(self.hparams.noise_level_img/255), 8)
            to_cat = torch.tensor([self.hparams.noise_level_img/255])
            if self.hparams.degradation_mode == 'SR':
                to_cat = to_cat.repeat(self.hparams.maxitr-40)
            else:
                to_cat = to_cat.repeat(self.hparams.maxitr-8)
            dpir_sigmas_ = torch.cat((dpir_sigmas_, to_cat), dim=0)
            dpir_sigmas=dpir_sigmas_.to(self.device)
            dpir_rhos = 0.23*((self.hparams.noise_level_img/255)**2)/(dpir_sigmas**2)
        if self.hparams.PnP_algo == 'FISTA':
            tk=1
            y = x
        if self.hparams.PnP_algo == 'adaPGD':
            thetak = 1
            lambdak = 1
            gradx = None
        while i < self.hparams.maxitr and torch.abs(diff_Psi)/torch.abs(Psi_old) > self.hparams.relative_diff_Psi_min and BFGSBreakFlag > 0:
            
            if self.hparams.inpainting_init :
                if i < self.hparams.n_init:
                    self.sigma_denoiser = 50
                else :
                    self.sigma_denoiser = self.hparams.sigma_denoiser
            else :
                self.sigma_denoiser = self.hparams.sigma_denoiser


            x_old = x
            Psi_old = Psi

            if self.hparams.PnP_algo == 'PGD':
                # Gradient step
                gradx = self.calculate_grad(x_old)
                z = x_old - self.hparams.lamb*gradx
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(z, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
                Dz = z - Dg
                Dx = Dz
                x = (1 - self.hparams.alpha) * z + self.hparams.alpha*Dz
                y = x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                F = self.calculate_F(x, z, g, img_tensor)

            elif self.hparams.PnP_algo == 'aPGD':
                # Gradient step
                q = (1-aPGD_alpha)*y + aPGD_alpha * x_old
                gradx = self.calculate_grad(q)
                z = x_old - aPGD_lambda*gradx
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(z, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
                x = z - gamma * Dg # use relaxed denoiser
                # Dx = Dz # useless
                y = (1 - aPGD_alpha) * y + aPGD_alpha*x # relax again
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                Psi = torch.tensor([self.calculate_F(x, z, g, img_tensor)]).to(self.device)
                F = Psi.item()
                diff_Psi = Psi-Psi_old

            elif self.hparams.PnP_algo == 'DRS':
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(x_old, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(x_old.double() - N.double(), p=2) ** 2)
                Dx = x_old - Dg
                y = (1 - self.hparams.alpha)*x_old + self.hparams.alpha*Dx
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    y = torch.clamp(y,0,1)
                # Proximal step
                z = self.calculate_prox(2*y-x_old)
                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                diff_Psi = Psi-Psi_old
                # Calculate Objective
                F = self.calculate_F(y, x, g, img_tensor)
                # Final step
                x = x_old + (z-y)

            elif self.hparams.PnP_algo == 'DRSdiff':

                # Proximal step
                y = self.calculate_prox(x_old)
                y2 = 2*y-x_old

                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(y2, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(y2.double() - N.double(), p=2) ** 2)
                Dx = y2 - Dg
                z = (1 - self.hparams.alpha) * y2 + self.hparams.alpha * Dx
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    z = torch.clamp(z, 0, 1)

                # Calculate Lyapunov
                Psi = self.calculate_lyapunov_DRS(y,z,x,g,img_tensor)
                diff_Psi = Psi-Psi_old
                # Calculate Objective
                F = self.calculate_F(y, x, g, img_tensor)
                # Final step
                x = x_old + (z-y)

            elif self.hparams.PnP_algo == 'BFGS':
                x_old = x
                
                # L_f = self.calculate_Lipschitz(x_old)
                #print(L_f)
                
                # Gradient step
                # Put lambdas in front of grads to make things consistent with PnP-PGD paper
                gradx = self.calculate_grad(x_old)
                flag = True
                flag_tau = False
                force_pass = False
                flag_override = False
                tau = 1
                T_gamma, R_gamma, Nx = utils_qn.TR_gamma(x_old, self.hparams.lamb*gradx, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True,alpha = self.hparams.alpha)
                grad_phi_gamma = R_gamma - gamma * self.hparams.lamb * self.calculate_Hessian(R_gamma)
                dk = searchdir_maker.compute_search(grad_phi_gamma, True)
                phi_gamma_x = utils_qn.calculate_phi_gamma(x_old, gradx, 
                                        self.calulate_data_term(x.double(),torch.Tensor(img).double().to('cuda')),
                                        self.denoiser_model,
                                        gamma, self.hparams.sigma_denoiser / 255., self.hparams.lamb, self.hparams.alpha)

                # print(torch.norm(grad_phi_gamma))
                try:
                    if torch.abs(phi_gamma_x_old - phi_gamma_x) < 5e-5:
                        BFGSBreakFlag = max(BFGSBreakFlag-1,0)
                    else:
                        BFGSBreakFlag = 5
                    if torch.abs(phi_gamma_x_old - phi_x) < 1e-6:
                        BFGSBreakFlag = max(BFGSBreakFlag-1,0)
                except:
                    pass
                if BFGSBreakFlag == 0:
                    flag = False
                    print("reached atol")
                # try:
                #     print(phi_gamma_x < phi_gamma_w)
                # except:
                #     print('a')
                flagGD = False
                flagDoNotUpdate = False
                while flag: # While gamma or tau is not OK
                    # gradx = self.calculate_grad(x_old)
                    # T_gamma, R_gamma = utils_qn.TR_gamma(x_old, self.hparams.lamb*gradx, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255.)
                    # grad_phi_gamma = R_gamma - gamma * self.calculate_Hessian(R_gamma)
                    # self.test_Hessian()
                    # raise Exception

                    #print(grad_phi_gamma.shape)
                    # Search direction
                    # dk = utils_qn.compute_searchdir(y_arr, s_arr, grad_phi_gamma, m)

                    # if flag_tau:
                    #     dk = -grad_phi_gamma
                    #     #dk = searchdir_maker.compute_search(grad_phi_gamma, True)
                    # elif flag_override:
                    #     dk = searchdir_maker.compute_search(grad_phi_gamma, True)
                    # else:
                    #     dk = searchdir_maker.compute_search(grad_phi_gamma)

                    # print(torch.tensordot(dk, grad_phi_gamma, dims=4))
                    # Armijo Line search
                    # TODO implement 
                    # Replacement: Secant condition

                    w = x + tau * dk
                    gradw = self.calculate_grad(w)
                    # Tw, Rw, Nw = utils_qn.TR_gamma(w, self.hparams.lamb*gradw, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True)
                    # Check sufficient descent

                    phi_gamma_w = utils_qn.calculate_phi_gamma(w, gradw, 
                        self.calulate_data_term(w.double(),torch.Tensor(img).double().to('cuda')),
                        self.denoiser_model,
                        gamma, self.hparams.sigma_denoiser / 255., self.hparams.lamb, self.hparams.alpha)

                    if phi_gamma_w <= phi_gamma_x:
                        flag = False
                    elif tau < 0.0001 and not flagGD:
                        gam = torch.tensordot(searchdir_maker.s_arr[-1], searchdir_maker.y_arr[-1], dims = 4) / torch.tensordot(searchdir_maker.y_arr[-1], searchdir_maker.y_arr[-1], dims = 4)
                        dk = -grad_phi_gamma * torch.abs(gam) # Just in case.
                        tau = 1
                        flagGD = True
                        
                    elif tau < 0.00001 and flagGD: # bfgs and gd step both fail.
                        tau = 0
                        flag = False
                        w = x
                        gradw = gradx
                        phi_gamma_w = phi_gamma_x 
                        flagDoNotUpdate = True
                    else:
                        tau = tau * 0.5 
                    # if phi_gamma_w <= phi_gamma_x + 1e-7:
                    #     flag = False
                    # elif tau < 1e-5:
                    #     flagDoNotUpdate = True
                    #     tau = 0
                    # else:
                    #     tau = tau * 0.5 
                        # print(phi_gamma_w, phi_gamma_x)


                    # if not flag:
                    #     Tw, Rw, Nw = utils_qn.TR_gamma(w, self.hparams.lamb*gradw, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True, alpha = self.hparams.alpha)

                    #     cond_LHS = self.hparams.lamb* self.calulate_data_term(Tw.double(),torch.Tensor(img).double().to('cuda'))
                    #     # cond_RHS = (self.hparams.lamb*self.calulate_data_term(x_old,torch.Tensor(img).to('cuda')) 
                    #     #             - gamma * torch.tensordot(self.hparams.lamb*gradx, R_gamma.double(), dims = len(x.shape))
                    #     #             + (1-Beta) * gamma / 2 * torch.linalg.norm(R_gamma)**2)
                    #     cond_RHS = (self.hparams.lamb*self.calulate_data_term(w,torch.Tensor(img).to('cuda')) 
                    #                 - gamma * torch.tensordot(self.hparams.lamb*gradw, R_gamma.double(), dims = len(x.shape))
                    #                 + (1-Beta) * gamma / 2 * torch.linalg.norm(Rw)**2)
                    #     cond = cond_LHS > cond_RHS
                    #     if cond:
                    #         gammaDecreaseFlag = gammaDecreaseFlag - 1

                    
                            # print("condition held - gamma is bad")
                        # if gammaDecreaseFlag == 0:
                        #     gamma = gamma * 0.95
                        #     gammaDecreaseFlag = 5
                        #     print("decreasing gamma")






                    # if cond or force_pass:
                    #     flag = False
                    # else:
                    #     if tau < 0.0001:
                    #         if not flag_tau and not flag_override:
                    #             tau = 1
                    #             flag_override = True
                    #             print("flagged", torch.tensordot(dk, grad_phi_gamma, dims=4))
                    #         elif not flag_tau:
                    #             tau = 1
                    #             flag_tau = True
                    #             print("flag tau")
                    #         else:
                    #             flag = False
                    #             force_pass = True
                    #             print("force pass")
                    #     tau = tau * 0.1

                if tau != 1:
                    print(i,tau)     
                # Sufficient descent is attained
                Tw, Rw, Nw = utils_qn.TR_gamma(w, self.hparams.lamb*gradw, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True, alpha = self.hparams.alpha)
                s = w - x_old
                grad_phi_gamma_w = Rw - gamma * self.hparams.lamb * self.calculate_Hessian(Rw)
                y = grad_phi_gamma_w - grad_phi_gamma
                if not flagDoNotUpdate:
                    searchdir_maker.push_ys(y, s)
                x = Tw

                
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                #foo = x - gamma*gradx
                
                F = phi_gamma_x.item()
                y = x
                z = w
                phi_gamma_x_old = phi_gamma_x
                phi_x = utils_qn.calculate_phi_x(self.calulate_data_term(x.double(),torch.Tensor(img).double().to('cuda')),
                        w, gradw, Tw, Nw, gamma, self.hparams.lamb, self.hparams.alpha)
                # print(phi_gamma_x.item())
                # print(phi_gamma_w.item())
            
            elif self.hparams.PnP_algo == 'BFGS2':
                x_old = x
                
                # L_f = self.calculate_Lipschitz(x_old)
                #print(L_f)
                
                # Gradient step
                # Put lambdas in front of grads to make things consistent with PnP-PGD paper
                gradx = self.calculate_grad(x_old)
                flag = True
                flag_tau = False
                force_pass = False
                flag_override = False
                tau = 1
                T_gamma, R_gamma, Nx = utils_qn.TR_gamma(x_old, self.hparams.lamb*gradx, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True,alpha = self.hparams.alpha)
                grad_phi_gamma = R_gamma - gamma * self.hparams.lamb * self.calculate_Hessian(R_gamma)
                dk = searchdir_maker.compute_search(grad_phi_gamma, True)
                phi_gamma_x = utils_qn.calculate_phi_gamma(x_old, gradx, 
                                        self.calulate_data_term(x.double(),torch.Tensor(img).double().to('cuda')),
                                        self.denoiser_model,
                                        gamma, self.hparams.sigma_denoiser / 255., self.hparams.lamb, self.hparams.alpha)
                # print(torch.norm(grad_phi_gamma))
                # try:
                #     if torch.abs(phi_gamma_x_old - phi_gamma_x) < 5e-5:
                #         BFGSBreakFlag = max(BFGSBreakFlag-1,0)
                #     else:
                #         BFGSBreakFlag = 5
                #     if torch.abs(phi_gamma_x_old - phi_x) < 1e-6:
                #         BFGSBreakFlag = max(BFGSBreakFlag-1,0)
                # except:
                #     pass
                # if BFGSBreakFlag == 0:
                #     flag = False
                #     print("reached atol")

                flagGD = False
                flagDoNotUpdate = False
                while flag: # While gamma or tau is not OK

                    w = x + tau * dk
                    gradw = self.calculate_grad(w)
                    # Tw, Rw, Nw = utils_qn.TR_gamma(w, self.hparams.lamb*gradw, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True)
                    # Check sufficient descent

                    phi_gamma_w = utils_qn.calculate_phi_gamma(w, gradw, 
                        self.calulate_data_term(w.double(),torch.Tensor(img).double().to('cuda')),
                        self.denoiser_model,
                        gamma, self.hparams.sigma_denoiser / 255., self.hparams.lamb, self.hparams.alpha)

                    if phi_gamma_w <= phi_gamma_x:
                        flag = False
                    elif tau < 0.0001 and not flagGD:
                        gam = torch.tensordot(searchdir_maker.s_arr[-1], searchdir_maker.y_arr[-1], dims = 4) / torch.tensordot(searchdir_maker.y_arr[-1], searchdir_maker.y_arr[-1], dims = 4)
                        dk = -grad_phi_gamma * torch.abs(gam) # Just in case.
                        tau = 1
                        flagGD = True
                        
                    elif tau < 0.00001 and flagGD: # bfgs and gd step both fail.
                        tau = 0
                        flag = False
                        w = x
                        gradw = gradx
                        phi_gamma_w = phi_gamma_x 
                        flagDoNotUpdate = True
                    else:
                        tau = tau * 0.5 

                if tau != 1:
                    print(i,tau)         
                # Sufficient descent is attained
                Tw, Rw, Nw = utils_qn.TR_gamma(w, self.hparams.lamb*gradw, self.denoiser_model, gamma, self.hparams.sigma_denoiser / 255., return_N = True, alpha = self.hparams.alpha)
                s = w - x_old
                grad_phi_gamma_w = Rw - gamma * self.hparams.lamb * self.calculate_Hessian(Rw)
                y = grad_phi_gamma_w - grad_phi_gamma
                if not flagDoNotUpdate:
                    searchdir_maker.push_ys(y, s)
                x = Tw

                
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                #foo = x - gamma*gradx
                
                F = phi_gamma_x.item()
                y = x
                z = w
                phi_gamma_x_old = phi_gamma_x
                phi_x = utils_qn.calculate_phi_x(self.calulate_data_term(x.double(),torch.Tensor(img).double().to('cuda')),
                        w, gradw, Tw, Nw, gamma, self.hparams.lamb, self.hparams.alpha)
                # print(phi_gamma_x.item())
                # print(phi_gamma_w.item())
                if i > 0 and torch.abs(phi_x - Psi_list[-1]) < 1e-8:
                    BFGSBreakFlag=0
            elif self.hparams.PnP_algo == 'DPIR' or self.hparams.PnP_algo == 'DPIR2':
                # HQS: prox_f follwed by denoiser
                tau = dpir_rhos[i].float().repeat(1, 1, 1, 1)
                # y = self.calculate_prox(x_old).float() # prox step
                y = utils_sr.prox_solution(x_old, self.FB, self.FBC, self.F2B, self.FBFy, tau, dpir_sf).float()
                foo = dpir_sigmas[i].float().repeat(1, 1, y.shape[2], y.shape[3])
                if x8:
                    y = augment_img_tensor4(y, i % 8)
                y_append = torch.cat((y, dpir_sigmas[i].float().repeat(1, 1, y.shape[2], y.shape[3])), dim=1)
                # z = dpir_model(y_append)

                z = test_mode(dpir_model, y_append, mode=2, refield=32, min_size=256, modulo=16)
                if x8:
                    if i % 8 == 3 or i % 8 == 5:
                        z = augment_img_tensor4(z, 8 - i % 8)
                    else:
                        z = augment_img_tensor4(z, i % 8)
                x = z    
                y = x
                # print(z.shape)
            elif self.hparams.PnP_algo == 'FISTA':
                # Gradient step
                tk1 = (1+np.sqrt(1+4*tk**2))/2
                grady = self.calculate_grad(y)
                z = y - self.hparams.lamb*grady
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(z, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
                Dz = z - Dg
                Dx = Dz
                x = (1 - self.hparams.alpha) * z + self.hparams.alpha*Dz
                y = x + (tk-1)/tk1 * (x-x_old)
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                F = self.calculate_F(x, z, g, img_tensor)
                tk = tk1
            elif self.hparams.PnP_algo == 'adaPGD':
                # Gradient step
                gradx_old = gradx
                gradx = self.calculate_grad(x_old)
                z = x_old - lambdak*gradx
                # Denoising step
                torch.set_grad_enabled(True)
                Dg, N = self.denoiser_model.calculate_grad(z, self.hparams.sigma_denoiser / 255.)
                torch.set_grad_enabled(False)
                Dg = Dg.detach()
                N = N.detach()
                g = 0.5 * (torch.norm(z.double() - N.double(), p=2) ** 2)
                Dz = z - Dg
                Dx = Dz
                x = (1 - self.hparams.alpha) * z + self.hparams.alpha*Dz
                y = x
                # Hard constraint
                if self.hparams.use_hard_constraint:
                    x = torch.clamp(x,0,1)
                # Calculate Objective
                F = self.calculate_F(x, z, g, img_tensor)
                if gradx_old is not None:
                    foo = torch.linalg.norm(z-x_old)/(2*torch.linalg.norm(gradx_old-gradx))
                    lambdak1 = np.min((np.sqrt(1+thetak)*lambdak,foo.item()))
                    lambdak = lambdak1
                    thetak = lambdak/lambdak1
                else:
                    lambdak1=1
                    thetak=100000

            else :
                print('algo not implemented')

            # Logging
            if extract_results:
                out_y = tensor2array(y.cpu())
                out_z = tensor2array(z.cpu())
                out_x = tensor2array(x.cpu())
                current_y_psnr = psnr(clean_img, out_y)
                current_z_psnr = psnr(clean_img, out_z)
                current_x_psnr = psnr(clean_img, out_x)
                if self.hparams.print_each_step:
                    print('iteration : ', i)
                    print('current y PSNR : ', current_y_psnr)
                    print('current z PSNR : ', current_z_psnr)
                    print('current x PSNR : ', current_x_psnr)
                    try:
                        print("current x phi_gamma: ", phi_gamma_x.item())
                        print("current w phi_gamma: ", phi_gamma_w.item())
                        print("current x phi: ", phi_x.item())
                        print("current tau: ", tau)
                    except:
                        pass
                y_list.append(out_y)
                x_list.append(out_x)
                z_list.append(out_z)
                # if self.hparams.PnP_algo != 'BFGS' and self.hparams.PnP_algo != 'BFGS2':
                #     y_list.append(out_y)
                #     x_list.append(out_x)
                #     z_list.append(out_z)
                # else:
                #     if BFGSBreakFlag>0:
                #         y_list.append(out_y)
                #         x_list.append(out_x)
                #         z_list.append(out_z)
                if self.hparams.PnP_algo == 'BFGS' and BFGSBreakFlag == 0:
                    y_list.pop()
                    x_list.pop()
                    z_list.pop()
                # Dx_list.append(tensor2array(Dx.cpu()))
                # Dg_list.append(torch.norm(Dg).cpu().item())
                # g_list.append(g.cpu().item())
                psnr_tab.append(current_x_psnr)
                F_list.append(F)
                if self.hparams.PnP_algo == 'BFGS':
                    Psi_list.append(torch.real(phi_x.cpu()))
            if self.hparams.PnP_algo == 'BFGS2':
                Psi_list.append(torch.real(phi_x.cpu()))
            # next iteration
            i += 1
            # print(diff_Psi, Psi_old)

        output_img = tensor2array(y.cpu())
        output_psnr = psnr(clean_img, output_img)
        output_psnrY = psnr(rgb2y(clean_img), rgb2y(output_img))

        if extract_results:
            # return output_img, output_psnr, output_psnrY, x_list, np.array(z_list), np.array(y_list), np.array(Dg_list), np.array(psnr_tab), np.array(Dx_list), np.array(g_list), np.array(F_list), np.array(Psi_list)
            # return output_img, output_psnr, output_psnrY, x_list, np.array(z_list), np.array(y_list), [], np.array(psnr_tab), [], [], np.array(F_list), np.array(Psi_list)
            return output_img, output_psnr, output_psnrY, x_list, np.array(z_list), np.array(y_list), [], np.array(psnr_tab), [], [], np.array(F_list), np.array(Psi_list)

        else:
            return output_img, output_psnr, output_psnrY

    def initialize_curves(self):

        self.conv = []
        self.PSNR = []
        self.F = []
        self.Psi = []

    def update_curves(self, x_list, psnr_tab, F_list, Psi_list):
        self.Psi.append(Psi_list)
        self.F.append(F_list)
        self.PSNR.append(psnr_tab)
        self.conv.append(np.array([(np.linalg.norm(x_list[k + 1] - x_list[k]) ** 2) for k in range(len(x_list) - 1)]) / np.sum(np.abs(x_list[0]) ** 2))
   
    def save_curves(self, save_path):

        import matplotlib
        matplotlib.rcParams.update({'font.size': 12})
        matplotlib.rcParams['lines.linewidth'] = 2

        plt.figure(1)
        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.PSNR)):
            plt.plot(self.PSNR[i], markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim([17.5,40.0])
        ax.set_xlabel("Iter")
        ax.set_ylabel("PSNR")
        # plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, 'PSNR.png'),bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'PSNR.pdf'),bbox_inches="tight")
        plt.figure(22)

        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.F)):
            plt.plot(self.F[i], markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlabel("Iter")
        ax.set_xlabel(r"$k$ (iter)")
        ax.set_ylabel(r"$\varphi_\gamma(x^k)$")

        # plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, 'phi_gamma_x.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'phi_gamma_x.pdf'), bbox_inches="tight")

        plt.figure(162)
        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.Psi)):
            plt.plot(self.Psi[i], markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlabel("Iter")
        ax.set_xlabel(r"$k$ (iter)")
        ax.set_ylabel(r"$\varphi(x^k)$")
        # plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, 'phi_x.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'phi_x.pdf'), bbox_inches="tight")


        try:
            plt.figure(164)
            fig, ax = plt.subplots()
            # ax.spines['right'].set_visible(False)
            # ax.spines['top'].set_visible(False)
            for i in range(len(self.Psi)):
                varphi = self.Psi[i]
                varphi_gamma = self.F[i]
                plt.plot(varphi[:-1] - varphi_gamma[1:], markevery=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # plt.legend()
            plt.semilogy()
            plt.grid()
            # ax.set_xlabel("Iter")
            ax.set_xlabel(r"$k$ (iter)")
            ax.set_ylabel(r"$(\varphi- \varphi_\gamma)(x^k)$")
            plt.savefig(os.path.join(save_path, 'moreau_difference.png'), bbox_inches="tight")
            plt.savefig(os.path.join(save_path, 'moreau_difference.pdf'), bbox_inches="tight")
        except:
            pass

        plt.figure(5)
        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.conv)):
            plt.plot(self.conv[i], '-o', markevery=10)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # plt.legend()
        plt.grid()
        plt.savefig(os.path.join(save_path, 'conv_log.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'conv_log.pdf'), bbox_inches="tight")
        self.conv2 = [[np.min(self.conv[i][:k]) for k in range(1, len(self.conv[i]))] for i in range(len(self.conv))]
        conv_rate = [self.conv2[i][0]*np.array([(1/k) for k in range(1,self.hparams.maxitr)]) for i in range(len(self.conv2))]
        conv_rate_2 = [self.conv2[i][0]*np.array([(1/k**2) for k in range(1,self.hparams.maxitr)]) for i in range(len(self.conv2))]
        plt.figure(6)
        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.conv)):
            plt.plot(self.conv2[i], '-', markevery=10)
        plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.plot(conv_rate_2[i], '--', color='b', label=r'$\mathcal{O}(\frac{1}{K^2})$')
        plt.semilogy()
        plt.legend()
        plt.grid()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlabel("Iter")
        # ax.set_ylabel(r"$\min_{i\leq k} ||x^{i+1} - x^i||^2$")
        ax.set_xlabel(r"$k$ (iter)")
        ax.set_ylabel(r"$\min_{i\leq k} ||x^{i+1} - x^i||^2/||x^0||^2$")
        if self.hparams.degradation_mode == 'SR':
            # ax.set_ylim(bottom=1e-3, top=1e-14)
            ax.set_ylim(bottom=1e-14, top=1e-3)
        else:
            # ax.set_ylim(bottom=1e-2, top=1e-12)
            ax.set_ylim(bottom=1e-12, top=1e-2)
        plt.savefig(os.path.join(save_path, 'conv_log2.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'conv_log2.pdf'), bbox_inches="tight")

        plt.figure(7)
        fig, ax = plt.subplots()
        # ax.spines['right'].set_visible(False)
        # ax.spines['top'].set_visible(False)
        for i in range(len(self.conv)):
            plt.plot(self.conv[i], '-', markevery=10)
        plt.plot(conv_rate[i], '--', color='red', label=r'$\mathcal{O}(\frac{1}{K})$')
        plt.plot(conv_rate_2[i], '--', color='b', label=r'$\mathcal{O}(\frac{1}{K^2})$')
        plt.semilogy()
        plt.legend()
        plt.grid()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # ax.set_xlabel("Iter")
        # ax.set_ylabel(r"$\min_{i\leq k} ||x^{i+1} - x^i||^2$")
        ax.set_xlabel(r"$k$ (iter)")
        ax.set_ylabel(r"$||x^{k+1} - x^k||^2/||x^0||^2$")
        if self.hparams.degradation_mode == 'SR':
            # ax.set_ylim(bottom=1e-3, top=1e-14)
            ax.set_ylim(bottom=1e-14, top=1e-3)
        else:
            # ax.set_ylim(bottom=1e-2, top=1e-12)
            ax.set_ylim(bottom=1e-12, top=1e-2)
        
        plt.savefig(os.path.join(save_path, 'conv_notmin.png'), bbox_inches="tight")
        plt.savefig(os.path.join(save_path, 'conv_notmin.pdf'), bbox_inches="tight")

    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--dataset_path', type=str, default='../datasets')
        parser.add_argument('--pretrained_checkpoint', type=str,default='../GS_denoising/ckpts/Prox-DRUNet.ckpt')
        parser.add_argument('--PnP_algo', type=str, default='PGD')
        parser.add_argument('--dataset_name', type=str, default='CBSD68')
        parser.add_argument('--sigma_denoiser', type=float)
        parser.add_argument('--sigma_k_denoiser', type=float)
        parser.add_argument('--noise_level_img', type=float, default=2.55)
        parser.add_argument('--sigma_multi', type=float, default=1.)
        parser.add_argument('--maxitr', type=int, default=1000)
        parser.add_argument('--alpha', type=float, default=1)
        parser.add_argument('--lamb', type=float)
        parser.add_argument('--gamma', type=float, default=0.9)
        parser.add_argument('--beta', type=float, default=0.)
        parser.add_argument('--n_images', type=int, default=68)
        parser.add_argument('--relative_diff_Psi_min', type=float, default=1e-8)
        parser.add_argument('--inpainting_init', dest='inpainting_init', action='store_true')
        parser.set_defaults(inpainting_init=False)
        parser.add_argument('--precision', type=str, default='simple')
        parser.add_argument('--n_init', type=int, default=10)
        parser.add_argument('--patch_size', type=int, default=256)
        parser.add_argument('--extract_curves', dest='extract_curves', action='store_true')
        parser.set_defaults(extract_curves=False)
        parser.add_argument('--extract_images', dest='extract_images', action='store_true')
        parser.set_defaults(extract_images=False)
        parser.add_argument('--print_each_step', dest='print_each_step', action='store_true')
        parser.set_defaults(print_each_step=False)
        parser.add_argument('--no_grad_matching', dest='grad_matching', action='store_false')
        parser.set_defaults(grad_matching=True)
        parser.add_argument('--no_data_term', dest='no_data_term', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--use_hard_constraint', dest='use_hard_constraint', action='store_true')
        parser.set_defaults(no_data_term=False)
        parser.add_argument('--full', dest='full', action='store_true')
        parser.set_defaults(full=False)
        return parser
