from typing import Callable, List
from jax import jit, grad, value_and_grad, random
from jax import custom_vjp, vmap, hessian, jacfwd, jacrev, jvp, vjp
from functools import partial
from jax.scipy import stats
from jax.tree_util import tree_map

import numpy as np 
import jax.numpy as jnp
import flax
import flax.linen as nn
import diffrax

def activation(x):
    """Antiderivative of Tanh."""
    absx = jnp.abs(x)
    return absx + jnp.log(1 + jnp.exp(-2 * absx))


def activation_prime(x):
    """Tanh."""
    return jnp.tanh(x)


def activation_2prime(x):
    """Derivative of Tanh"""
    return 1 / jnp.square(jnp.cosh(x))

    
class _Phi(nn.Module):
    """Old, using AD"""
    input_dim: int

    # ResNet Args
    hidden_dim: int
    resnet_depth: int
    rank: int
    resnet_stepsize: float = 1.
    resnet_activation: Callable = activation
    resnet_kernel_init: Callable = nn.initializers.he_normal()
    resnet_bias_init: Callable = nn.initializers.constant(0.1)

    # Initilizers
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.constant(0.1)
    constant_init: Callable = nn.initializers.normal()

    
    @nn.compact
    def __call__(self, inputs):
        A = self.param('kernel', self.kernel_init, (self.rank, self.input_dim)) 
        b = self.param('bias', self.bias_init, (self.input_dim,))
        c = self.param('const', self.constant_init, (1,))
        theta = self.param('resnet_bias', self.bias_init, (self.hidden_dim,))
        
        # ResNet
        z = nn.Dense(
                features=self.hidden_dim,
                kernel_init=self.resnet_kernel_init,
                bias_init=self.resnet_bias_init,
            )(inputs)
        z = self.resnet_activation(z)
        
        for i in range(self.resnet_depth - 1):
            out = nn.Dense(
                features=self.hidden_dim,
                kernel_init=self.resnet_kernel_init,
                bias_init=self.resnet_bias_init,
            )(z)
            z += self.resnet_stepsize * self.resnet_activation(out)
        # z is now the output of the ResNet    
            
        
        A_dot_s = jnp.dot(A, inputs)
        
        return jnp.dot(theta, z) + 0.5 * jnp.dot(A_dot_s, A_dot_s) + jnp.dot(b, inputs) + c
    
    
class Phi(nn.Module):
    input_dim: int

    # ResNet Args
    hidden_dim: int
    resnet_depth: int
    rank: int
    resnet_stepsize: float = 1.
    resnet_activation: Callable = activation
    resnet_activation_prime: Callable = activation_prime
    resnet_activation_2prime: Callable = activation_2prime
    resnet_kernel_init: Callable = nn.initializers.he_normal()
    resnet_bias_init: Callable = nn.initializers.constant(0.1)

    # Initilizers
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.constant(0.1)
    constant_init: Callable = nn.initializers.normal()
    
    
    def setup(self):
        self.A = self.param('kernel', self.kernel_init, (self.rank, self.input_dim)) 
        self.b = self.param('bias', self.bias_init, (self.input_dim,))
        self.c = self.param('const', self.constant_init, (1,))
        self.w = self.param('resnet_bias', self.bias_init, (self.hidden_dim,))
        
        self.dense_layers = [nn.Dense(
                                features=self.hidden_dim,
                                kernel_init=self.resnet_kernel_init,
                                bias_init=self.resnet_bias_init,
                             ) 
                             for i in range(self.resnet_depth)]
    

    def __call__(self, inputs):
        # ResNet
        z = self.dense_layers[0](inputs)
        z = self.resnet_activation(z)
        
        for i in range(1, self.resnet_depth):
            out = self.dense_layers[i](z)
            z += self.resnet_stepsize * self.resnet_activation(out)
        # z is now the output of the ResNet    
            
        A_dot_s = jnp.dot(self.A, inputs)
        
        return jnp.dot(self.w, z) + 0.5 * jnp.dot(A_dot_s, A_dot_s) + jnp.dot(self.b, inputs) + self.c    
    
    
    def grad(self, inputs):
        """Computes grad(phi) for the given inputs."""
        # forward pass, save intermediate activations
        act = self.dense_layers[0](inputs)
        activations = [act]
        u = self.resnet_activation(act)
        
        for submodule in self.dense_layers[1:]:
            act = submodule(u)
            activations.append(act)
            u += self.resnet_stepsize * self.resnet_activation(act)
            
        # compute the gradients     
        z = self.w
        for act, submodule in zip(reversed(activations[1:]), reversed(self.dense_layers[1:])):
            out = self.resnet_activation_prime(act)
            K_T = submodule.variables['params']['kernel']
            
            z += self.resnet_stepsize * jnp.dot(K_T, out * z)
              
        K_T = self.dense_layers[0].variables['params']['kernel'] 

        # grad_phi is grad(N)w + A'As + b = z0 + A'As + b
        return jnp.dot(K_T, self.resnet_activation_prime(activations[0]) * z) + jnp.dot(self.A.T, jnp.dot(self.A, inputs)) + self.b
        
    
    def grad_trhess(self, inputs):
        """Computes grad(phi) and Tr[Hess(Phi)] for the given input."""
        # forward pass, save intermediate activations
        act = self.dense_layers[0](inputs)
        activations = [act]
        u = self.resnet_activation(act)
        
        for submodule in self.dense_layers[1:]:
            act = submodule(u)
            activations.append(act)
            u += self.resnet_stepsize * self.resnet_activation(act)
                
        # compute the gradients    
        z = self.w
        zs = [z]
        for act, submodule in zip(reversed(activations[1:]), reversed(self.dense_layers[1:])):
            out = self.resnet_activation_prime(act)
            K_T = submodule.variables['params']['kernel']
            
            z += self.resnet_stepsize * jnp.dot(K_T, out * z)
            zs.append(z)
            
        # compute z0 = grad(phi) * w    
        K_T = self.dense_layers[0].variables['params']['kernel'] 
        zs.append(jnp.dot(K_T, self.resnet_activation_prime(activations[0]) * z)) # append z0
        # zs is now [z_M+1, z_M, ..., z_0]

        zs.reverse()  
        
        # compute t0
        K = self.dense_layers[0].variables['params']['kernel'].T # K0
        t = jnp.dot(
            (self.resnet_activation_2prime(activations[0]) * zs[1]).T, 
            jnp.sum(jnp.power(K[:, :self.input_dim - 1], 2), axis=1)
        )
        
        # compute J, this will be updated in every iteration
        J = self.resnet_activation_prime(activations[0])[:, None] * K[:, :self.input_dim - 1]
        
        # respective 1st elements of iterables below: z2, K1 * u0 + b1, (K1, b1)
        for z, act, submodule in zip(zs[2:], activations[1:], self.dense_layers[1:]):
            KJ = jnp.dot(submodule.variables['params']['kernel'].T, J) 
            t += self.resnet_stepsize * jnp.dot(
                (self.resnet_activation_2prime(act) * z).T, 
                jnp.sum(jnp.power(KJ, 2), axis=1)
            )
            J += self.resnet_stepsize * self.resnet_activation_prime(act)[:, None] * KJ
            
        # nicht speichern?
        AtimesE = self.A[:, :self.input_dim - 1]
        
        #return grad_s(phi), tr(hess(phi))
        return (zs[0] + jnp.dot(self.A.T, jnp.dot(self.A, inputs)) + self.b, 
                t + jnp.trace(jnp.dot(AtimesE.T, AtimesE)))
        
        # compute tr(E'A'AE) only once
    
    
def value_and_jacfwd(f, x):
    pushfwd = partial(jvp, f, (x,))
    basis = jnp.eye(x.size, dtype=x.dtype)
    y, jac = vmap(pushfwd, out_axes=(None, 1))((basis,))
    return y, jac    
    
    
    
class OTF():
    def __init__(
        self, 
        input_dim, 
        hidden_dim: int,
        resnet_depth: int,
        rank: int,
        key: random.PRNGKey, 
        phi: Callable,
        num_blocks: int,
        alpha1,
        alpha2,
        prior_type='gaussian', 
        prior_args=None, 
        t0 = 0.,
        t1 = 1.,
        num_steps = 10,
        resnet_stepsize = 1.
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.resnet_depth = resnet_depth
        self.rank = rank
        self.num_blocks = num_blocks
        self.resnet_stepsize = resnet_stepsize

        self.t0 = t0
        self.t1 = t1
        self.num_steps = num_steps
        self.dt0 = (self.t1 - self.t0) / (self.num_blocks * self.num_steps)
        
        self.key, param_init_key = random.split(key=key, num=2)

        self.alpha1 = alpha1
        self.alpha2 = alpha2

        self.funcs = []
        self.params = []
        for i in range(num_blocks):
            func = phi(
                input_dim=input_dim + 1,  # +1 for the time dim
                hidden_dim=hidden_dim,
                resnet_depth=resnet_depth,
                resnet_stepsize=resnet_stepsize,
                rank=rank,
            )
            param = func.init(param_init_key, inputs=jnp.empty(input_dim + 1))

            self.funcs.append(func) 
            self.params.append(param) 
            param_init_key = random.fold_in(param_init_key, i)
        

        self.prior_type = prior_type
        self.prior_args = prior_args
        

    def _gaussian_diag_cov_log(self, x, mean=None, sigma=None):
        k = x.shape[-1] # input dim
        
        if mean is None:
            mean = jnp.zeros(k)

        if sigma is None:
            sigma = jnp.ones(k)

        sigma_sq = jnp.square(sigma)
        return jnp.sum(-jnp.square(x - mean) / (2 * sigma_sq) - 0.5 * jnp.log(2 * jnp.pi * sigma_sq), axis=-1)


    def _gaussian_diag_cov(self, x, mean=None, sigma=None):
        k = x.shape[-1] # input dim

        if mean is None:
            mean = jnp.zeros(k)

        if sigma is None:
            sigma = jnp.ones(k)
        
        sigma_sq = jnp.square(sigma)
        return jnp.prod(jnp.exp(-jnp.square(x - mean) / (2 * sigma_sq)) / jnp.sqrt(2 * jnp.pi * sigma_sq), axis=-1)

    
    def log_pdf_base_dist(self, x):
        if self.prior_type=='gaussian':
            if self.prior_args is not None:
                mean, sigma = self.prior_args['mean'], self.prior_args['sigma']
                return self._gaussian_diag_cov_log(x, mean=mean, sigma=sigma)
            else:
                return self._gaussian_diag_cov_log(x)

        elif self.prior_type=='mollified_uniform':
            minval, maxval, sigma = self.prior_args['minval'], self.prior_args['maxval'], self.prior_args['sigma']
            densities_uni = 1 / (maxval - minval) * (stats.norm.cdf((maxval - x) / sigma) - stats.norm.cdf((minval - x) / sigma)) 
            return jnp.sum(jnp.log(densities_uni)) #this is for single sample since we use vmap anyways, axis=1)

        elif self.prior_type=='gaussian_mixture':
            mixture_logits, means, sigmas = self.prior_args['mixture_logits'], self.prior_args['means'], self.prior_args['sigmas']
            mixture_weights = softmax(mixture_logits)

            density = 0.
            for mixture_weight, mean, sigma in zip(mixture_weights, means, sigmas):
                density += mixture_weight * self._gaussian_diag_cov(x, mean=mean, sigma=sigma)
                
            return jnp.log(density)
            
        else: raise ValueError(f'Invalid prior_type: {self.prior_type}')
        
        
    @staticmethod
    def forward_dynamics(t, y, args):
        phi, params = args
        grad_x = phi.apply(params, jnp.hstack((y, jnp.array(t))), method='grad')[:y.shape[-1]]

        return -grad_x
        

    @staticmethod
    def backward_dynamics(t, y, args):
        phi, params = args
        y, _, _, _ = y   
        
        grad_s, trhess = phi.apply(params, jnp.hstack((y, jnp.array(t))), method='grad_trhess')
        grad_x, grad_t = jnp.split(grad_s.squeeze(), [y.shape[-1]])

        norm_grad_x_over2 = 0.5 * jnp.dot(grad_x, grad_x)
        
        return (-grad_x,                                            # -grad(Phi)      
                -trhess,                                            # -Tr(hess(Phi))
                norm_grad_x_over2,                                  # 0.5 * || grad(Phi) || ** 2
                jnp.abs(grad_t.squeeze() - norm_grad_x_over2))      # | d/dtPhi - 0.5 * || grad(Phi) ||** 2 | 
    
    
    @staticmethod
    def reduced_backward_dynamics(t, y, args):
        phi, params = args
        y, _ = y   
        
        grad_s, trhess = phi.apply(params, jnp.hstack((y, jnp.array(t))), method='grad_trhess')
        grad_x, grad_t = jnp.split(grad_s.squeeze(), [y.shape[-1]])
        
        return (-grad_x, -trhess)


    def sample_base_dist(self, n):
        self.key, subkey = random.split(self.key)
        if self.prior_type=='gaussian':
            if self.prior_args is not None:
                mean, sigma = self.prior_args['mean'], self.prior_args['sigma']
                sample = mean + random.normal(self.key, (n, self.input_dim)) * sigma
                
            else:
                sample = random.normal(self.key, (n, self.input_dim)) 

        elif self.prior_type=='mollified_uniform':
            minval, maxval, sigma = self.prior_args['minval'], self.prior_args['maxval'], self.prior_args['sigma']
            sample = random.uniform(key=self.key, shape=(n, self.input_dim), minval=minval, maxval=maxval)
            
            self.key, subkey = random.split(self.key)
            sample += sigma * random.normal(key=self.key, shape=(n, self.input_dim))

        elif self.prior_type=='gaussian_mixture':
            mixture_logits, means, sigmas = self.prior_args['mixture_logits'], self.prior_args['means'], self.prior_args['sigmas']
            cat_sample = random.categorical(self.key, logits=mixture_logits, shape=(n,))
            sample = []
            for idx, m in zip(*jnp.unique(cat_sample, return_counts=True)):
                self.key, subkey = random.split(self.key)
                mean, sigma = means[idx], sigmas[idx]
                sample.append(mean + random.normal(self.key, (m, self.input_dim)) * sigma)
                
            sample = jnp.vstack(sample)

        else: raise ValueError(f'Invalid prior_type: {self.prior_type}')
        
        return sample


    def propagate(self, data, solver_steps=None, params=None):
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
            
        def _wrap(sample):
            y = sample
            
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t1
            t_end = self.t1 - delta
            
            for param, func in zip(params, self.funcs):
                args=(func, param)
                term = diffrax.ODETerm(self.forward_dynamics)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, -delta / solver_steps, y, args)
                (y,) = sol.ys
                
                t_start = t_end
                t_end -= delta
            return y
            
        return vmap(_wrap, 0, 0)(data)
    
    
    def propagate_with_steps(self, data, solver_steps=None, params=None, intermed_y=False):
        if params is None:
            params = self.params

        if solver_steps is None:
            solver_steps = self.num_steps
            
        kwargs = {
            'saveat': diffrax.saveat.SaveAt(t0=True, steps=True),
            'max_steps': solver_steps
        } if intermed_y else {}

        def _wrap(sample):
            container = [sample]
            y = sample
            
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t1
            t_end = self.t1 - delta
            
            for param, func in zip(params, self.funcs):
                args=(func, param)
                term = diffrax.ODETerm(self.forward_dynamics)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, -delta / solver_steps, y, args, **kwargs)
                y = sol.ys[-1]
                container.append(sol.ys)
                
                t_start = t_end
                t_end -= delta
                
            return container
        return vmap(_wrap, 0, 0)(data) 

    
    def sample(self, n, solver_steps=None, params=None):
        if params is None:
            params = self.params
        x = self.sample_base_dist(n)
        
        if solver_steps is None:
            solver_steps = self.num_steps

        def _wrap(sample):
            y = sample
            
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t1
            t_end = self.t1 - delta
            
            for param, func in zip(params, self.funcs):
                args=(func, param)
                term = diffrax.ODETerm(self.forward_dynamics)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, -delta / solver_steps, y, args)
                (y,) = sol.ys
                
                t_start = t_end
                t_end -= delta
                
            return y
        
        return vmap(_wrap, 0, 0)(x)  
    
    
    def sample_with_steps(self, n, solver_steps=None, params=None, intermed_y=False):
        if params is None:
            params = self.params
        x = self.sample_base_dist(n)
        
        if solver_steps is None:
            solver_steps = self.num_steps

        kwargs = {
            'saveat': diffrax.saveat.SaveAt(t0=True, steps=True),
            'max_steps': solver_steps
        } if intermed_y else {}

        def _wrap(sample):
            container = [sample]
            y = sample
            
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t1
            t_end = self.t1 - delta
            
            for param, func in zip(params, self.funcs):
                args=(func, param)
                term = diffrax.ODETerm(self.forward_dynamics)
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, -delta / solver_steps, y, args, **kwargs)
                y = sol.ys[-1]
                container.append(sol.ys)
                
                t_start = t_end
                t_end -= delta
                
            return container
        return vmap(_wrap, 0, 0)(x)      
        

    def log_pdf_and_preimage(self, datapoint, solver_steps=None, params=None, return_preimg=False):
        """
        Sample is of shape (input_dim,)
        """
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
        
        term = diffrax.ODETerm(self.backward_dynamics)
        solver = diffrax.Tsit5()
        
        l, L, R = 0., 0., 0.
        y = (datapoint, l, L, R)
        
        delta = (self.t1 - self.t0) / self.num_blocks
        t_start = self.t0
        t_end = self.t0 + delta
        
        for param, func in zip(reversed(params), reversed(self.funcs)):
            args=(func, param)

            # Note: default is adjoint=discretise-then-optimise 
            sol = diffrax.diffeqsolve(
                term, solver, t_start, t_end, delta / solver_steps, y, args
            )
            (y,), (l,), (L,), (R,) = sol.ys
            y = (y, l, L, R)
            t_start = t_end
            t_end += delta 
        
        (y,), (l,), (L,), (R,) = sol.ys
        if return_preimg: 
            return -(l + self.log_pdf_base_dist(y)), L, R, y
            
        return -(l + self.log_pdf_base_dist(y)), L, R
    
    
    def log_pdf_and_preimage_reduced(self, datapoint, solver_steps=None, params=None, return_preimg=False):
        """
        Sample is of shape (input_dim,)
        """
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
        
        term = diffrax.ODETerm(self.reduced_backward_dynamics)
        solver = diffrax.Tsit5()
        
        l = 0.
        y = (datapoint, l)
        
        delta = (self.t1 - self.t0) / self.num_blocks
        t_start = self.t0
        t_end = self.t0 + delta
        
        for param, func in zip(reversed(params), reversed(self.funcs)):
            args=(func, param)

            # Note: default is adjoint=discretise-then-optimise 
            sol = diffrax.diffeqsolve(
                term, solver, t_start, t_end, delta / solver_steps, y, args
            )
            (y,), (l,) = sol.ys
            y = (y, l)
            t_start = t_end
            t_end += delta 
        
        (y,), (l,) = sol.ys
        if return_preimg: 
            return -(l + self.log_pdf_base_dist(y)), y
            
        return -(l + self.log_pdf_base_dist(y))
       
    
    def _loss(self, params, batch, solver_steps=None):
        if solver_steps is None:
            solver_steps = self.num_steps
            
        CLR = vmap(self.log_pdf_and_preimage,  (0, None, None, None), 0)(batch, solver_steps, params, False)
        CLR = tree_map(jnp.mean, CLR)
        return (self.alpha1 * CLR[0] + CLR[1] + self.alpha2 * CLR[2]) / self.alpha1
    
    
    def nll(self, params, batch, solver_steps=None):
        if solver_steps is None:
            solver_steps = self.num_steps
            
        C = vmap(self.log_pdf_and_preimage_reduced,  (0, None, None, None), 0)(batch, solver_steps, params, False)
        return jnp.mean(C)
    
    
    def _gaussiankernel(self, X, Y):
        """Returns N x M Gaussian Kernel Matrix"""
        return jnp.exp(-0.5 * jnp.square(jnp.linalg.norm(X[None,:,:] - Y[:,None,:], axis=2).T))
    
    
    def metrics(self, params, batch, solver_steps=None, normal_batch=None):
        if solver_steps is None:
            solver_steps = self.num_steps
            
        # compute NLL and preimages
        C, y = vmap(self.log_pdf_and_preimage_reduced,  (0, None, None, None), 0)(batch, solver_steps, params, True)
        C = jnp.mean(C)
        
        # inverse error
        x = self.propagate(y, solver_steps, params)
        inv_error = jnp.mean(jnp.linalg.norm(batch - x, axis=1))
        
        # maximum mean discrepancy
        x = self.propagate(normal_batch, solver_steps, params)
        mmd = jnp.mean(self._gaussiankernel(batch,batch)) + jnp.mean(self._gaussiankernel(x,x)) - 2 * jnp.mean(self._gaussiankernel(x,batch))
        return C, inv_error, mmd