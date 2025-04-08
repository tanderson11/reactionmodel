import numpy as np
from scipy.special import binom

try:
    from numba import jit as numbajit
except ImportError:
    numbajit = None

def construct_propensity_function(k, kinetic_order_matrix, jit=False):
    if jit:
        if numbajit is None:
            raise ImportError("numba is not installed, cannot use jit. Try installing with extra dependencies.")
        kinetic_order_matrix = np.asarray(kinetic_order_matrix)

        return _jit_construct_propensity_function(k, kinetic_order_matrix)
    return _construct_propensity_function(k, kinetic_order_matrix)

def _construct_propensity_function(k, kinetic_order_matrix):
    homogeneous = isinstance(k, np.ndarray)

    def calculate_propensities(t, y):
        if homogeneous:
            k_of_t = k
        else:
            k_of_t = k(t, y)
        # product along column in kinetic order matrix
        # with states raised to power of involvement
        # multiplied by rate constants == propensity
        # dimension of y is expanded to make it a column vector
        return np.prod(binom(np.expand_dims(y, axis=1), kinetic_order_matrix), axis=0) * k_of_t
    return calculate_propensities

def _jit_construct_propensity_function(k, kinetic_order_matrix):
    homogeneous = isinstance(k, np.ndarray)
    @numbajit(nopython=True)
    def jit_calculate_propensities(t, y):
        # Remember, we want total number of distinct combinations * k === rate.
        # we want to calculate (y_i kinetic_order_ij) (binomial coefficient)
        # for each species i and each reaction j
        # sadly, inside a numba C function, we can't avail ourselves of scipy's binom,
        # so we write this little calculator ourselves
        # the for loops will be optimized out by numba
        intensity_power = np.zeros_like(kinetic_order_matrix)
        for i in range(0, kinetic_order_matrix.shape[0]):
            for j in range(0, kinetic_order_matrix.shape[1]):
                if y[i] < kinetic_order_matrix[i][j]:
                    intensity_power[i][j] = 0.0
                elif y[i] == kinetic_order_matrix[i][j]:
                    intensity_power[i][j] = 1.0
                else:
                    intensity = 1.0
                    for x in range(0, kinetic_order_matrix[i][j]):
                        intensity *= (y[i] - x) / (x+1)
                    intensity_power[i][j] = intensity
        if not homogeneous:
            k_evaled = k(t, y)
        else:
            k_evaled = k
        product_down_columns = np.ones(len(k_evaled))
        for i in range(0, len(y)):
            product_down_columns = product_down_columns * intensity_power[i]
        return product_down_columns * k_evaled
    return jit_calculate_propensities