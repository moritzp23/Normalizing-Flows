from typing import Callable, List
from jax import jit, grad, value_and_grad, random
from jax import custom_vjp, vmap, hessian, jacfwd, jacrev, jvp, vjp
from jax.tree_util import tree_leaves
from functools import partial
from jax.scipy import stats

import numpy as np 
import jax.numpy as jnp
import flax
import flax.linen as nn
import diffrax



class ConcatSquash(nn.Module):
    features: int

    def setup(self):
        self.lin1 = nn.Dense(features=self.features,
                             kernel_init=nn.initializers.he_normal(),
                             bias_init=nn.initializers.constant(0.01))
        self.lin2 = nn.Dense(features=self.features,
                             kernel_init=nn.initializers.he_normal(),
                             bias_init=nn.initializers.constant(0.01))
        self.lin3 = nn.Dense(features=self.features,
                             use_bias=False,
                             kernel_init=nn.initializers.he_normal(),
                             bias_init=nn.initializers.constant(0.01))
        

    def __call__(self, t, y):
        return self.lin1(y) * nn.sigmoid(self.lin2(t)) + self.lin3(t)


class f_theta(nn.Module):
    hidden_dim: int
    out_dim: int
    depth: int

    def setup(self):
        layers = []
        if self.depth == 0:
            layers.append(
                ConcatSquash(features=self.out_dim)
            )
        else:
            layers.append(
                ConcatSquash(features=self.hidden_dim)
            )
            for i in range(self.depth - 1):
                layers.append(
                    ConcatSquash(features=self.hidden_dim)
                )
            layers.append(
                ConcatSquash(features=self.out_dim)
            )
            self.layers = layers


    def __call__(self, t, y):
        t = jnp.asarray(t)[None]
        for layer in self.layers[:-1]:
            y = layer(t, y)
            y = nn.tanh(y)
        y = self.layers[-1](t, y)
        return y


class CNF():
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        out_dim, 
        depth, 
        num_blocks,
        key: random.PRNGKey, 
        f_theta_cls: Callable = f_theta,
        prior_type='gaussian', 
        prior_args=None, 
        exact_logp=False,
        num_steps=10
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.num_blocks = num_blocks

        self.t0 = 0.0
        self.t1 = 1.0
        self.num_steps = num_steps
        self.dt0 = (self.t1 - self.t0) / (self.num_blocks * self.num_steps)


        self.exact_logp = exact_logp
        
        self.key, param_init_key = random.split(key=key, num=2)

        self.funcs = []
        self.params = []
        for i in range(num_blocks):
            func = f_theta_cls(hidden_dim=hidden_dim, out_dim=out_dim, depth=depth)
            param = func.init(param_init_key, t=0., y=jnp.empty(input_dim))

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
    def approx_logp_wrapper(t, y, args):
        """TODO: Fix this"""
        eps, func = args
        y, _ = y
        fn = lambda y: func(t, y)
        f, vjp_fn = vjp(fn, y)
        (eps_dfdy,) = vjp_fn(eps)
        logp = jnp.sum(eps_dfdy * eps)
        return f, logp

    
    @staticmethod
    def exact_logp_wrapper(t, y, args):
        eps, func = args
        y, _ = y
        fn = lambda y: func(t, y)

        basis = jnp.eye(y.shape[0], dtype=y.dtype)                                          # standard basis
        f, jac = vmap(jvp, in_axes=(None, None, 0), out_axes=(None, 1))(fn, (y,), (basis,)) # vectorize over the different basis vectors
        
        logp = jnp.trace(jac)
        return f, logp


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
            t_start = self.t0
            t_end = self.t0 + delta
            
            for param, func in zip(params, self.funcs):
                term = diffrax.ODETerm(lambda t, y, args: func.apply(param, t=t, y=y))
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, delta / solver_steps, y)
                (y,) = sol.ys
                
                t_start = t_end
                t_end += delta
                
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
            t_start = self.t0
            t_end = self.t0 + delta
            
            for param, func in zip(params, self.funcs):
                term = diffrax.ODETerm(lambda t, y, args: func.apply(param, t=t, y=y))
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, delta / solver_steps, y, **kwargs)
                y = sol.ys[-1]
                container.append(sol.ys)
                
                t_start = t_end
                t_end += delta
                
            return container
            
        return vmap(_wrap, 0, 0)(data)

    
    def sample(self, n, solver_steps=None, params=None):
        """
        Draws a single sample. TODO: implement this with vmap
        """
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
            
        x = self.sample_base_dist(n)

        def _wrap(sample):
            y = sample
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t0
            t_end = self.t0 + delta
            for param, func in zip(params, self.funcs):
                term = diffrax.ODETerm(lambda t, y, args: func.apply(param, t=t, y=y))
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, delta / solver_steps, y)
                (y,) = sol.ys
                
                t_start = t_end
                t_end += delta
                
            return y
        return vmap(_wrap, 0, 0)(x)


    def sample_with_steps(self, n, solver_steps=None, params=None, intermed_y=False):
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
            
        x = self.sample_base_dist(n)

        kwargs = {
            'saveat': diffrax.saveat.SaveAt(t0=True, steps=True),
            'max_steps': solver_steps
        } if intermed_y else {}

        def _wrap(sample):
            container = [sample]
            y = sample
            delta = (self.t1 - self.t0) / self.num_blocks
            t_start = self.t0
            t_end = self.t0 + delta
            for param, func in zip(params, self.funcs):
                term = diffrax.ODETerm(lambda t, y, args: func.apply(param, t=t, y=y))
                solver = diffrax.Tsit5()
                sol = diffrax.diffeqsolve(term, solver, t_start, t_end, delta / solver_steps, y, **kwargs)
                y = sol.ys[-1]
                container.append(sol.ys)
                
                t_start = t_end
                t_end += delta
                
            return container
        return vmap(_wrap, 0, 0)(x)      
        

    def log_pdf_and_preimage(self, datapoint, solver_steps=None, params=None, return_preimg=False):
        """
        Sample is of shape (input_dim,)
        
        Compared to the other methods, this has an additional 'solver_steps' argument, since this is typically 
        jitted so simply modifying self.num_steps wont work without doing recompilation.
        """
        if params is None:
            params = self.params
            
        if solver_steps is None:
            solver_steps = self.num_steps
        
        if self.exact_logp:
            term = diffrax.ODETerm(self.exact_logp_wrapper)
        else:
            term = diffrax.ODETerm(self.approx_logp_wrapper)
        solver = diffrax.Tsit5()

        #self.key, eps_key = random.split(self.key, num=2)
        #eps = random.normal(eps_key, datapoint.shape)
        if self.exact_logp: 
            eps = np.random.normal(size=datapoint.shape)
        else: 
            eps = None
        
        delta_log_likelihood = 0.0
        y = datapoint
        
        # TODO: das als IVP machen
        delta = (self.t1 - self.t0) / self.num_blocks
        t_start = self.t1
        t_end = self.t1 - delta

        for param, func in zip(reversed(params), reversed(self.funcs)):
            y = (y, delta_log_likelihood)
            args = (eps, partial(func.apply, param))

            # Note: default is adjoint=discretise-then-optimise 
            sol = diffrax.diffeqsolve(
                term, solver, t_start, t_end, -delta / solver_steps, y, args # (eps,func) are passed to term
            )
            (y,), (delta_log_likelihood,) = sol.ys
            
            t_start = t_end
            t_end -= delta

        if return_preimg: 
            return delta_log_likelihood + self.log_pdf_base_dist(y), y
            
        return delta_log_likelihood + self.log_pdf_base_dist(y)
       
    
    def _loss(self, params, batch, solver_steps=None):
        if solver_steps is None:
            solver_steps = self.num_steps
        
        log_pdf = vmap(self.log_pdf_and_preimage,  (0, None, None, None), 0)(batch, solver_steps, params, False)
        return -jnp.mean(log_pdf)


    def _loss_l2(self, params, batch, alpha, solver_steps=None):
        if solver_steps is None:
            solver_steps = self.num_steps
        
        log_pdf = vmap(self.log_pdf_and_preimage,  (0, None, None, None), 0)(batch, solver_steps, params, False)
        loss = -jnp.mean(log_pdf)
        
        for layer_param in params:
            for param in tree_leaves(layer_param["params"]):
                if len(param.shape) == 1:
                    continue
                loss += alpha * (param ** 2).sum()

        return loss
    

    def _gaussiankernel(self, X, Y):
        """Returns N x M Gaussian Kernel Matrix"""
        return jnp.exp(-0.5 * jnp.square(jnp.linalg.norm(X[None,:,:] - Y[:,None,:], axis=2).T))
    
    
    def metrics(self, params, batch, solver_steps=None, normal_batch=None):
        if solver_steps is None:
            solver_steps = self.num_steps
            
        # compute NLL and preimages
        C, y = vmap(self.log_pdf_and_preimage,  (0, None, None, None), 0)(batch, solver_steps, params, True)
        C = -jnp.mean(C)
        
        # inverse error
        x = self.propagate(y, solver_steps, params)
        inv_error = jnp.mean(jnp.linalg.norm(batch - x, axis=1))
        
        # maximum mean discrepancy
        x = self.propagate(normal_batch, solver_steps, params)
        mmd = jnp.mean(self._gaussiankernel(batch,batch)) + jnp.mean(self._gaussiankernel(x,x)) - 2 * jnp.mean(self._gaussiankernel(x,batch))
        return C, inv_error, mmd