import jax.numpy as jnp
import flax.linen as nn

from jax.tree_util import tree_leaves
from jax import jit, grad, random, jacfwd, vmap, jvp, value_and_grad
from typing import Callable
from jax.nn import softmax
from functools import partial
from jax.scipy import stats


class NVP_l(nn.Module):
    hidden_dim: int
    out_dim: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1,
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'), 
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1, 
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'),
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1/jnp.sqrt(2), 
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'), 
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.tanh(x)
        return x
    
    
class NVP_dropl(nn.Module):
    hidden_dim: int
    out_dim: int
    dropout_proba: float
    
    @nn.compact
    def __call__(self, x, training):
        x = nn.Dense(features=self.hidden_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1,
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'), 
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.Dropout(rate=self.dropout_proba, deterministic=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.hidden_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1, 
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'),
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.Dropout(rate=self.dropout_proba, deterministic=not training)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.out_dim, 
                     kernel_init=nn.initializers.variance_scaling(scale=1/np.sqrt(2), 
                                                                  mode='fan_avg',
                                                                  distribution='truncated_normal'), 
                     bias_init=nn.initializers.constant(0.01))(x)
        x = nn.tanh(x)
        return x


class RealNVP():
    def __init__(
        self, 
        NVP_net: Callable, 
        num_blocks: int,
        key: random.PRNGKey, 
        input_dim: int, 
        hidden_dim: int, 
        prior_type='gaussian', 
        prior_args=None, 
        use_dropout=False,
        dropout_proba=None
    ):
        self.num_blocks = num_blocks
        self.layers = []
        self.params = []
        self.flips = []
        self.dropout_key, param_init_key, dropout_init_key = random.split(key=key, num=3)
        self.input_dim = input_dim
        self.prior_type = prior_type
        self.prior_args = prior_args
        self.key = key
        
        self.use_dropout = use_dropout
        if self.use_dropout:
            rng_dict = {'params': param_init_key, 'dropout': dropout_init_key}
            self.training = True 
            
            self.call_kwargs = {'training': self.training,
                                'rngs': {'dropout': self.dropout_key}}
            setup_kwargs = {'dropout_proba': dropout_proba}
        
        else:
            rng_dict = {'params': param_init_key}
            self.call_kwargs = {}
            setup_kwargs = {}

        flip = False
        for i in range(self.num_blocks):
            dim = self.input_dim // 2
            param_init_key = random.fold_in(param_init_key, i)
            dropout_init_key = random.fold_in(dropout_init_key, i)
            layer = NVP_net(hidden_dim=hidden_dim, out_dim=self.input_dim, **setup_kwargs)   
            param = layer.init(rng_dict, jnp.empty(dim), **self.call_kwargs)

            self.layers.append(layer) 
            self.params.append(param) 
            self.flips.append(flip)
            flip = not flip
                

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
        return jnp.prod(
            jnp.exp(-jnp.square(x - mean) / (2 * sigma_sq)) / jnp.sqrt(2 * jnp.pi * sigma_sq),
            axis=-1
        )

    
    def log_pdf_base_dist(self, x):
        if self.prior_type=='gaussian':
            if self.prior_args is not None:
                mean, sigma = self.prior_args['mean'], self.prior_args['sigma']
                return self._gaussian_diag_cov_log(x, mean=mean, sigma=sigma)
            else:
                return self._gaussian_diag_cov_log(x)

        elif self.prior_type=='mollified_uniform':
            minval = self.prior_args['minval']
            maxval = self.prior_args['maxval']
            sigma = self.prior_args['sigma']
            
            densities_uni = 1 / (maxval - minval) * (stats.norm.cdf((maxval - x) / sigma) - 
                                                     stats.norm.cdf((minval - x) / sigma)) 
            return jnp.sum(jnp.log(densities_uni), axis=1)

        elif self.prior_type=='gaussian_mixture':
            mixture_logits = self.prior_args['mixture_logits']
            means = self.prior_args['means']
            sigmas = self.prior_args['sigmas']
            
            mixture_weights = softmax(mixture_logits)

            density = 0.
            for mixture_weight, mean, sigma in zip(mixture_weights, means, sigmas):
                density += mixture_weight * self._gaussian_diag_cov(x, mean=mean, sigma=sigma)
                
            return jnp.log(density)
            

        else: raise ValueError(f'Invalid prior_type: {self.prior_type}')



    def sample_base_dist(self, n):
        self.key, subkey = random.split(self.key)
        if self.prior_type=='gaussian':
            if self.prior_args is not None:
                mean, sigma = self.prior_args['mean'], self.prior_args['sigma']
                sample = mean + random.normal(self.key, (n, self.input_dim)) * sigma
                
            else:
                sample = random.normal(self.key, (n, self.input_dim)) 

        elif self.prior_type=='mollified_uniform':
            minval = self.prior_args['minval']
            maxval = self.prior_args['maxval']
            sigma = self.prior_args['sigma']
            sample = random.uniform(key=self.key, shape=(n, self.input_dim), minval=minval, maxval=maxval)
            
            self.key, subkey = random.split(self.key)
            sample += sigma * random.normal(key=self.key, shape=(n, self.input_dim))

        elif self.prior_type=='gaussian_mixture':
            mixture_logits = self.prior_args['mixture_logits']
            means = self.prior_args['means']
            sigmas = self.prior_args['sigmas']
            
            cat_sample = random.categorical(self.key, logits=mixture_logits, shape=(n,))
            sample = []
            for idx, m in zip(*jnp.unique(cat_sample, return_counts=True)):
                self.key, subkey = random.split(self.key)
                mean, sigma = means[idx], sigmas[idx]
                sample.append(mean + random.normal(self.key, (m, self.input_dim)) * sigma)
                
            sample = jnp.vstack(sample)

        else: raise ValueError(f'Invalid prior_type: {self.prior_type}')
        
        return sample


    def _nvp_forward(self, x, layer_callable, params, flip):
        n = x.shape[-1]
        d = n//2
        indices = jnp.arange(n)
        first, second = indices[:d], indices[d:]
        x1, x2 = x.take(first, axis=-1), x.take(second, axis=-1)
        if flip:
            x2, x1 = x1, x2
        s = layer_callable(params, x1, **self.call_kwargs)
        shift, log_scale = jnp.split(s, 2, axis=1)
        
        y2 = x2 * jnp.exp(log_scale) + shift
        if flip:
            x1, y2 = y2, x1
        y = jnp.concatenate([x1, y2], axis=-1)
        return y
    
    
    def _nvp_inverse(self, y, layer_callable, params, flip):
        n = y.shape[-1]
        d = n//2
        indices = jnp.arange(n)
        first, second = indices[:d], indices[d:]
        y1, y2 = y.take(first, axis=-1), y.take(second, axis=-1)
        if flip:
            y1, y2 = y2, y1
        s = layer_callable(params, y1, **self.call_kwargs)
        shift, log_scale = jnp.split(s, 2, axis=-1)
        x2 = (y2 - shift) * jnp.exp(-log_scale)
        if flip:
            y1, x2 = x2, y1
        x = jnp.concatenate([y1, x2], axis=-1)
        return x

    
    def sample(self, n, params=None):
        if params is None:
            params = self.params
        x = self.sample_base_dist(n)
        return self.forward_pass(x, params)
    
    
    def sample_stepwise(self, n, params=None):
        if params is None:
            params = self.params
            
        x = self.sample_base_dist(n)
        steps = [x]
        
        
        for i in range(len(self.layers)):
            x = self._nvp_forward(x, self.layers[i].apply, params[i], self.flips[i])
            steps.append(x)
        return steps
        

    def forward_pass(self, x, params):
        for i in range(len(self.layers)):
            x = self._nvp_forward(x, self.layers[i].apply, params[i], self.flips[i])
        return x

    
    def backward_pass(self, y, params):
        x = y
        for i in reversed(range(len(self.layers))):
            x = self._nvp_inverse(x, self.layers[i].apply, params[i], self.flips[i])
        return x


    def value_and_jacfwd(self, f, x, kwargs):
        """
        One could think of two other methods to get the jacobian of f evaluated at x: J_f(x) and f evaluated 
        at x: f(x) at the same time:
            (1) Tinkering something using jacfwd and has_aux (like wrapping f to return (f(x), f(x)) and treat 
                the second arg as aux
            (2) Just calling f and jacfwd seperately
        
        This however, is not ideal, since jacfwd basically executes the same code from below under the hood, 
        and just doesnt return y.
        """
        partial_f = partial(f, **kwargs)            # function f with all args except x frozen
        # pushfwd = partial(jvp, partial_f, (x,))   # function that takes tangent vectors v and returns (f(x), J_f(x) @ v), 
        basis = jnp.eye(x.shape[0], dtype=x.dtype)                 # standard basis
        y, jac = vmap(jvp, in_axes=(None, None, 0), out_axes=(None, 1))(partial_f, (x,), (basis,)) # vectorize over the different basis vectors
        return y, jac

    
    def value_and_jacfwd_batch(self, f, x, kwargs):
        """ Vectorize value_and_jacfwd along the first axis of x."""
        return vmap(self.value_and_jacfwd, (None, 0, None), (0,0))(f, x, kwargs)
        

    def log_pdf_and_preimage (self, data, params=None, return_preimg=False):
        if len(data.shape) == 1:
            data = data.reshape(1,-1)
        
        if params is None:
            params = self.params
            
        preimages, jacobians = self.value_and_jacfwd_batch(self.backward_pass, data, {'params': params})
        
        if return_preimg: 
            return self.log_pdf_base_dist(preimages) + jnp.linalg.slogdet(jacobians)[1], preimages
        
        return self.log_pdf_base_dist(preimages) + jnp.linalg.slogdet(jacobians)[1]
       
    
    def _loss(self, params, batch):
        preimages, jacobians = self.value_and_jacfwd_batch(self.backward_pass, batch, {'params': params})
        return -jnp.mean(self.log_pdf_base_dist(preimages) + jnp.linalg.slogdet(jacobians)[1])
        

    def _loss_l2(self, params, batch, alpha=0.001):
        preimages, jacobians = self.value_and_jacfwd_batch(self.backward_pass, batch, {'params': params})
        loss = -jnp.mean(self.log_pdf_base_dist(preimages) + jnp.linalg.slogdet(jacobians)[1])

        for layer_param in params:
            for param in tree_leaves(layer_param["params"]):
                if len(param.shape) == 1:
                    continue
                loss += alpha * (param ** 2).mean()
            
        return loss
    
    
    def _gaussiankernel(self, X, Y):
        """Returns N x M Gaussian Kernel Matrix"""
        return jnp.exp(-0.5 * jnp.square(jnp.linalg.norm(X[None,:,:] - Y[:,None,:], axis=2).T))
    
    
    def metrics(self, params, batch, normal_batch):
        # compute NLL and preimages
        C, y = self.log_pdf_and_preimage(batch, params, True)
        C = -jnp.mean(C)
        
        # inverse error
        x = self.forward_pass(y, params)
        inv_error = jnp.mean(jnp.linalg.norm(batch - x, axis=1))
        
        # maximum mean discrepancy
        x = self.forward_pass(normal_batch, params)
        mmd = jnp.mean(self._gaussiankernel(batch,batch)) + jnp.mean(self._gaussiankernel(x,x)) - 2 * jnp.mean(self._gaussiankernel(x,batch))
        return C, inv_error, mmd