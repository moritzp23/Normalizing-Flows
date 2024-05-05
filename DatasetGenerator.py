import jax.numpy as jnp
import numpy as np
from jax import random
import matplotlib.pyplot as plt
from functools import partial


def _E_of_x(one_over_m, s):
    b1 = bool(s)
    b2 = b1 * 30 + 2 * s
    
    # (x ** 2) / 2
    sqo2 = lambda x: jnp.power(x, 2) / 2
    
    return one_over_m * (( (sqo2(140)      - sqo2(20))       * 30 ) +
                         ( (sqo2(95)       - sqo2(65))       * 170) +
                         ( (sqo2(170 + s)  - sqo2(140 + s))  * 200) +
                         ( (sqo2(220 + s)  - sqo2(170 + s))  * 30 ) +
                         ( (sqo2(250 + s)  - sqo2(220 + s))  * 200) +
                         ( (sqo2(340 + b2) - sqo2(250 + b2)) * 30 ) +
                         ( (sqo2(340 + b2) - sqo2(250 + b2)) * 30 ) +
                    b1 * ( (sqo2(250 + b2) - sqo2(220 + b2)) * 200) )


def _E_of_y(one_over_m, s):
    b1 = bool(s)
    b2 = b1 * 30 + 2 * s
    
    # (x ** 2) / 2
    sqo2 = lambda x: jnp.power(x, 2) / 2    
    
    return one_over_m * (( 120 * (sqo2(130) - sqo2(100)) ) +
                         ( 30  * (sqo2(300) - sqo2(130)) ) +
                         ( 30  * (sqo2(300) - sqo2(100)) ) +
                         ( 50  * (sqo2(300) - sqo2(270)) ) +
                         ( 30  * (sqo2(300) - sqo2(100)) ) +
                         ( 90  * (sqo2(130) - sqo2(100)) ) +
                         ( 90  * (sqo2(300) - sqo2(270)) ) +
                    b1 * ( 30  * (sqo2(300) - sqo2(100)) ) )


def _E_of_xsq(one_over_m, s):
    b1 = bool(s)
    b2 = b1 * 30 + 2 * s
    
    # (x ** 3) / 3
    cbo3 = lambda x: jnp.power(x, 3) / 3 
    
    return one_over_m * (( (cbo3(140)      - cbo3(20))       * 30 ) +
                         ( (cbo3(95)       - cbo3(65))       * 170) +
                         ( (cbo3(170 + s)  - cbo3(140 + s))  * 200) +
                         ( (cbo3(220 + s)  - cbo3(170 + s))  * 30 ) +
                         ( (cbo3(250 + s)  - cbo3(220 + s))  * 200) +
                         ( (cbo3(340 + b2) - cbo3(250 + b2)) * 30 ) +
                         ( (cbo3(340 + b2) - cbo3(250 + b2)) * 30 ) +
                    b1 * ( (cbo3(250 + b2) - cbo3(220 + b2)) * 200) )


def _E_of_ysq(one_over_m, s):
    b1 = bool(s)
    b2 = b1 * 30 + 2 * s
    
    # (x ** 3) / 3
    cbo3 = lambda x: jnp.power(x, 3) / 3    
    
    return one_over_m * (( 120 * (cbo3(130) - cbo3(100)) ) +
                         ( 30  * (cbo3(300) - cbo3(130)) ) +
                         ( 30  * (cbo3(300) - cbo3(100)) ) +
                         ( 50  * (cbo3(300) - cbo3(270)) ) +
                         ( 30  * (cbo3(300) - cbo3(100)) ) +
                         ( 90  * (cbo3(130) - cbo3(100)) ) +
                         ( 90  * (cbo3(300) - cbo3(270)) ) +
                    b1 * ( 30  * (cbo3(300) - cbo3(100)) ) )


def make_tuc_letters(key, training_size, validation_size, test_size, spacing, dtype="float32"):
    dataset_size = training_size + validation_size + test_size
    b1 = bool(spacing)

    # Letter T
    m1 = (140 - 20) * (130 - 100)   
    m2 = (95 - 65) * (300 - 130) 

    # Letter U
    m3 = (170 - 140) * (300 - 100) 
    m4 = (220 - 170) * (300 - 270)  
    m5 = (250 - 220) * (300 - 100) 

    # Letter C
    m6 = (340 - 250) * (130 - 100) 
    m7 = (340 - 250) * (300 - 270) 
    m8 = (250 - 220) * (300 - 100) * b1

    m = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
    samplers = {
        1: partial(
            random.uniform,
            minval=jnp.array([20, 100]),
            maxval=jnp.array([140, 130]),
            dtype=dtype,
        ),
        2: partial(
            random.uniform,
            minval=jnp.array([65, 130]),
            maxval=jnp.array([95, 300]),
            dtype=dtype,
        ),
        3: partial(
            random.uniform,
            minval=jnp.array([140 + spacing, 100]),
            maxval=jnp.array([170 + spacing, 300]),
            dtype=dtype,
        ),
        4: partial(
            random.uniform,
            minval=jnp.array([170 + spacing, 270]),
            maxval=jnp.array([220 + spacing, 300]),
            dtype=dtype,
        ),
        5: partial(
            random.uniform,
            minval=jnp.array([220 + spacing, 100]),
            maxval=jnp.array([250 + spacing, 300]),
            dtype=dtype,
        ),
        6: partial(
            random.uniform,
            minval=jnp.array([250 + bool(spacing) * 30 + 2 * spacing, 100]),
            maxval=jnp.array([340 + bool(spacing) * 30 + 2 * spacing, 130]),
            dtype=dtype,
        ),
        7: partial(
            random.uniform,
            minval=jnp.array([250 + bool(spacing) * 30 + 2 * spacing, 270]),
            maxval=jnp.array([340 + bool(spacing) * 30 + 2 * spacing, 300]),
            dtype=dtype,
        ),
        8: partial(
            random.uniform,
            minval=jnp.array([220 + bool(spacing) * 30 + 2 * spacing, 100]),
            maxval=jnp.array([250 + bool(spacing) * 30 + 2 * spacing, 300]),
            dtype=dtype,
        ),
    }
    intsample = random.choice(
        key, 
        a=jnp.array([1, 2, 3, 4, 5, 6, 7, 8]), 
        p=jnp.array([m1, m2, m3, m4, m5, m6, m7, m8]) / m, shape=(dataset_size,)
    )
    nums, counts = jnp.unique(intsample, return_counts=True,)
    keys = random.split(key, num=len(nums))
    
    sample = []
    
    for i in range(len(nums)):
        n,k,c = nums[i].item(), keys[i], counts[i]
        sample.append(samplers.get(n)(key=k, shape=(c,2)))
        
        
    mu_x, mu_y = _E_of_x(1/m, spacing), _E_of_y(1/m, spacing)
    E_xsq, E_ysq = _E_of_xsq(1/m, spacing), _E_of_ysq(1/m, spacing)
    
    MEAN_VEC = jnp.array([mu_x, mu_y])
    
    # Note the factor -1 in the y-component since the letters are upside down
    STD_VEC = jnp.array([jnp.power(E_xsq - jnp.power(mu_x, 2), 0.5), -jnp.power(E_ysq - jnp.power(mu_y, 2), 0.5)])
    print(MEAN_VEC)
    print(STD_VEC)
    
    sample = (jnp.vstack(sample) - MEAN_VEC) / STD_VEC
    sample = random.permutation(key, sample, axis=0, independent=False)
    
    return sample[:training_size, :], sample[training_size:training_size+validation_size, :], sample[training_size+validation_size:, :]


def make_tuc_letters_tr(key, training_size, spacing, dtype="float32"):
    return make_tuc_letters(key, training_size, 0, 0, spacing, dtype)[0]


def get_letter_density(spacing):
    b1 = bool(spacing)

    # Letter T
    m1 = (140 - 20) * (130 - 100)   
    m2 = (95 - 65) * (300 - 130) 

    # Letter U
    m3 = (170 - 140) * (300 - 100) 
    m4 = (220 - 170) * (300 - 270)  
    m5 = (250 - 220) * (300 - 100) 

    # Letter C
    m6 = (340 - 250) * (130 - 100) 
    m7 = (340 - 250) * (300 - 270) 
    m8 = (250 - 220) * (300 - 100) * b1

    m = m1 + m2 + m3 + m4 + m5 + m6 + m7 + m8
    
    mu_x, mu_y = _E_of_x(1/m, spacing), _E_of_y(1/m, spacing)
    E_xsq, E_ysq = _E_of_xsq(1/m, spacing), _E_of_ysq(1/m, spacing)
    
    MEAN_VEC = jnp.array([mu_x, mu_y])
    
    # Note the factor -1 in the y-component since the letters are upside down
    STD_VEC = jnp.array([jnp.power(E_xsq - jnp.power(mu_x, 2), 0.5), -jnp.power(E_ysq - jnp.power(mu_y, 2), 0.5)])
    
    # standardization is a change of variables, change in density can be computed via abs(det(M))
    return jnp.abs(STD_VEC[0] * STD_VEC[1]) / m


def _make_tuc_letters(key, training_size, validation_size, test_size, spacing, dtype="float32", eps=0.02):
    """Deprecated"""
    dataset_size = training_size + validation_size + test_size
    mesh = np.zeros((600,600), dtype=dtype)

    # Letter T
    mesh[20:140, 100:130] = 1
    mesh[65:95, 120:300] = 1
    
    # Letter U
    mesh[140 + spacing:170 + spacing, 100:300] = 1
    mesh[170 + spacing:220 + spacing, 270:300] = 1
    mesh[220 + spacing:250 + spacing, 100:300] = 1
    
    # Letter C
    mesh[220 + bool(spacing) * 30 + 2 * spacing:340 + bool(spacing) * 30 + 2 * spacing, 100:130] = 1
    mesh[220 + bool(spacing) * 30 + 2 * spacing:340 + bool(spacing) * 30 + 2 * spacing, 270:300] = 1
    mesh[220 + bool(spacing) * 30 + 2 * spacing:250 + bool(spacing) * 30 + 2 * spacing, 100:300] = 1
    
    index = np.argwhere(mesh == 1)
    
    x = (index[:, 0] - index[:, 0].mean()) / index[:, 0].std()
    y = -(index[:, 1] - index[:, 1].mean()) / index[:, 1].std()
    coordinates = np.stack((x,y),axis=1)
    
    index_2 = random.randint(key, minval=0, maxval=len(coordinates), shape=(dataset_size, ))
    
    tuc = jnp.array(coordinates[index_2,:], dtype=dtype) + eps * random.normal(key, shape=(dataset_size, 2), dtype=dtype)
    return tuc[:training_size], tuc[training_size:training_size+validation_size], tuc[training_size+validation_size:]


def _make_tuc_letters_tr(key, training_size, spacing, dtype="float32", eps=0.02):
    return _make_tuc_letters(key, training_size, 0, 0, spacing, dtype, eps)[0]
    

def make_checkerboard(k, training_size, validation_size, test_size, dtype="float32"):
    dataset_size = training_size + validation_size + test_size
    k1, k2, k3 = random.split(k, 3)
    x1 = random.uniform(k1, (dataset_size, ), dtype) * 4 - 2
    x2_ = random.uniform(k2, (dataset_size, ), dtype) - random.randint(k3, (dataset_size, ), 0, 2) * 2
    x2 = x2_ + (np.floor(x1) % 2)
    x = jnp.concatenate([x1[:, None], x2[:, None]], 1) * 2
    return x[:training_size], x[training_size:training_size+validation_size], x[training_size+validation_size:]


def make_checkerboard_tr(k, training_size, dtype="float32"):
    return make_checkerboard(k, training_size, 0, 0, dtype)[0]


def make_tuc_logo(k, training_size, validation_size, test_size, img_path='tuc_logo.png', dtype="float32", eps=0.02):
    dataset_size = training_size + validation_size + test_size
    im = plt.imread(img_path).sum(axis=-1)
    index = np.argwhere(im != 0.)
    
    x = (index[:, 0] - index[:, 0].mean()) / index[:, 0].std()
    y = (index[:, 1] - index[:, 1].mean()) / index[:, 1].std()
    coordinates = np.stack((y,-x),axis=1)
    
    index_2 = random.randint(k, minval=0, maxval=len(coordinates), shape=(dataset_size, ))
    
    tuc = jnp.array(coordinates[index_2,:], dtype=dtype) + eps * random.normal(k, shape=(dataset_size, 2), dtype=dtype)
    return tuc[:training_size], tuc[training_size:training_size+validation_size], tuc[training_size+validation_size:]
    
    
def make_tuc_logo_tr(k, training_size, img_path='tuc_logo.png', dtype="float32", eps=0.02):
    return make_tuc_logo(k, training_size, 0, 0, img_path, dtype, eps)[0]
    