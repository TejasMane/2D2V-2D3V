import numpy as np
import arrayfire as af
from scipy.fftpack import fftfreq
from numpy.fft import fft2, ifft2


def fft_poisson(rho, dx, dy ):
    """
    FFT solver which returns the value of electric field. This will only work
    when the system being solved for has periodic boundary conditions.

    Parameters:
    -----------
    rho : The 1D/2D density array obtained from calculate_density() is passed to this
          function.

    dx  : Step size in the x-grid

    dy  : Step size in the y-grid.Set to None by default to avoid conflict with the 1D case.

    Output:
    -------
    E_x, E_y : Depending on the dimensionality of the system considered, either both E_x, and
               E_y are returned or E_x is returned.
    """
    print('rho.dims() is ', rho.dims())
    rho_temp = rho[0: -1, 0: -1]
    
    k_x = af.to_array(fftfreq(rho_temp.shape[1], dx))
    k_x = af.Array.as_type(k_x, af.Dtype.c64)
    k_y = af.to_array(fftfreq(rho_temp.shape[0], dy))
    k_x = af.tile(af.reorder(k_x), rho_temp.shape[0], 1)
    k_y = af.tile(k_y, 1, rho_temp.shape[1])
    k_y = af.Array.as_type(k_y, af.Dtype.c64)

    rho_hat       = fft2(rho_temp)
    rho_hat = af.to_array(rho_hat)
    potential_hat = af.constant(0, rho_temp.shape[0], rho_temp.shape[1], dtype=af.Dtype.c64)

    potential_hat       = (1/(4 * np.pi**2 * (k_x * k_x + k_y * k_y))) * rho_hat
    potential_hat[0, 0] = 0

    potential_hat = np.array(potential_hat)

    E_x_hat = -1j * 2 * np.pi * np.array(k_x) * potential_hat
    E_y_hat = -1j * 2 * np.pi * np.array(k_y) * potential_hat
    
    E_x = (ifft2(E_x_hat)).real
    E_y = (ifft2(E_y_hat)).real

    E_x = af.to_array(E_x)
    E_y = af.to_array(E_y)
    
    # Applying periodic boundary conditions
        
    E_x = af.join(0, E_x, E_x[0, :])
    E_x = af.join(1, E_x, E_x[:, 0])
    E_y = af.join(0, E_y, E_y[0, :])
    E_y = af.join(1, E_y, E_y[:, 0])   
    
    E_x[-1, -1] = E_x[0, 0].copy()
    E_y[-1, -1] = E_y[0, 0].copy()
    
    af.eval(E_x, E_y)
    return(E_x, E_y)