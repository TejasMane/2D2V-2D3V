import numpy as np
import arrayfire as af


""" Equations for fdtd (variation along x and y)"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B = dBz/dz


def fdtd( Ex, Ey, Bz, Lx, Ly, ghost_cells, Jx, Jy, dt):

    '''
    function fdtd( Ex, Ey, Bz, Lx, Ly, ghost_cells, Jx, Jy, dt)
    -----------------------------------------------------------------------  
    Input variables: Jx_Yee, number_of_electrons, w_p

        Jy_Yee: This is an array containing the currents deposited on Yee lattice.

        number_of_electrons: Number of macroparticles taken in the domain.
        
        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------  
    returns: Jy_norm_centered

        Jy_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''   
    
    forward_row     = af.Array([1, -1, 0])
    forward_column  = af.Array([1, -1, 0])
    backward_row    = af.Array([0, 1, -1])
    backward_column = af.Array([0, 1, -1])
    identity        = af.Array([0, 1, 0] )

    """ Number of grid points in the field's domain """

    (x_number_of_points,  y_number_of_points) = Bz.dims()

    """ number of grid zones calculated from the input fields """

    Nx = x_number_of_points - 2*ghost_cells-1
    Ny = y_number_of_points - 2*ghost_cells-1

    """ local variables for storing the input fields """

    Bz_local = Bz.copy()
    Ex_local = Ex.copy()
    Ey_local = Ey.copy()

    """Enforcing periodic BC's"""

    Bz_local = periodic_ghost(Bz_local, ghost_cells)

    Ex_local = periodic_ghost(Ex_local, ghost_cells)

    Ey_local = periodic_ghost(Ey_local, ghost_cells)

    """ Setting division size and time steps"""

    dx = np.float(Lx / (Nx))
    dy = np.float(Ly / (Ny))

    """ defining variable for convenience """

    dt_by_dx = dt / (dx)
    dt_by_dy = dt / (dy)


    """  Updating the Electric fields using the current too   """

    Ex_local += dt_by_dy * (af.signal.convolve2_separable(backward_row, identity, Bz_local)) - (Jx) * dt

    # dEx/dt = + dBz/dy

    Ey_local += -dt_by_dx * (af.signal.convolve2_separable(identity, backward_column, Bz_local)) - (Jy) * dt

    # dEy/dt = - dBz/dx

    """  Implementing periodic boundary conditions using ghost cells  """

    Ex_local = periodic_ghost(Ex_local, ghost_cells)

    Ey_local = periodic_ghost(Ey_local, ghost_cells)

    """  Updating the Magnetic field  """

    Bz_local += - dt_by_dx * (af.signal.convolve2_separable(identity, forward_column, Ey_local))  + dt_by_dy * (af.signal.convolve2_separable(forward_row, identity, Ex_local))

    # dBz/dt = - ( dEy/dx - dEx/dy )

    #Implementing periodic boundary conditions using ghost cells

    Bz_local = periodic_ghost(Bz_local, ghost_cells)

    af.eval(Bz_local, Ex_local, Ey_local)

    return Ex_local, Ey_local, Bz_local



def periodic_ghost(field, ghost_cells):

    '''
    function periodic_ghost(field, ghost_cells)
    -----------------------------------------------------------------------
    Input variables: field, ghost_cells

      field: An 2 dimensinal array representing an input field.(columns--->x, rows---->y)

      ghost_cells: Number of ghost cells taken in the domain

    -----------------------------------------------------------------------
    returns: field
      This function returns the modified field with appropriate values assigned to the ghost nodes to 
      ensure periodicity in the field.
    '''

    len_x = field.dims()[1]
    len_y = field.dims()[0]


    field[ 0 : ghost_cells, :]            = field[len_y -1 - 2 * ghost_cells\
                                                  : len_y -1 - 1 * ghost_cells, :\
                                                 ]

    field[ :, 0 : ghost_cells]            = field[:, len_x -1 - 2 * ghost_cells\
                                                  : len_x -1 - 1 * ghost_cells\
                                                 ]

    field[len_y - ghost_cells : len_y, :] = field[ghost_cells + 1:\
                                                  2 * ghost_cells + 1, :\
                                                 ]

    field[:, len_x - ghost_cells : len_x] = field[: , ghost_cells + 1\
                                                  : 2 * ghost_cells + 1\
                                                 ]

    af.eval(field)

    return field