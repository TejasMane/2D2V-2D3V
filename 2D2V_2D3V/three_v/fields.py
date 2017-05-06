import numpy as np
import arrayfire as af


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
    
    # determining the dimensions of the input field in both x and y directions
    len_x = field.dims()[1]
    len_y = field.dims()[0]
    
    # Assigning the values to the ghost nodes.
    
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





""" Equations for mode 1 FDTD"""

# dEz/dt = dBy/dx - dBx/dy
# dBx/dt = -dEz/dy
# dBy/dt = +dEz/dx
# div_B  = dBx/dx + dBy/dy

""" Equations for mode 2 FDTD"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B  = dBz/dz

"""
Notes for periodic boundary conditions:
for [0, length_domain_x] domain use periodic BC's such that last point in the physical domain coincides with the first point
for [0, length_domain_x) domain use periodic BC's such that the ghost point after the last physical point coincides with the first
physical point
"""


""" Alignment of the spatial grids for the fields(Convention chosen)

# This is the convention which will be used in the matrix representation

positive y axis -------------> going down
positive x axis -------------> going right

Now the fields aligned in x and y direction along with the following grids:


Ez  = (x_center, y_center ) 0, dt, 2dt, 3dt...
Bx  = (x_center, y_top    ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...
By  = (x_right, y_center  ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...

Ex  = (x_right, y_center  ) 0, dt, 2dt, 3dt...
Ey  = (x_center, y_top    ) 0, dt, 2dt, 3dt...
Bz  = (x_right, y_top     ) -0.5dt, 0.5dt, 1.5dt, 2.5dt...

rho = (x_center, y_center )  # Not needed here

Jx  = (x_right, y_center  ) 0.5dt, 1.5dt, 2.5dt...
Jy  = (x_center, y_top    ) 0.5dt, 1.5dt, 2.5dt...
Jz  = (x_center, y_center ) 0.5dt, 1.5dt, 2.5dt...
"""


""" Equations for mode 1 fdtd (variation along x and y)"""

# dEz/dt = dBy/dx - dBx/dy
# dBx/dt = -dEz/dy
# dBy/dt = +dEz/dx
# div_B = dBx/dx + dBy/dy

def mode1_fdtd( Ez, Bx, By, length_domain_x, length_domain_y, ghost_cells, Jx, Jy, Jz, dt):

    forward_row     = af.Array([1, -1, 0])
    forward_column  = af.Array([1, -1, 0])
    backward_row    = af.Array([0, 1, -1])
    backward_column = af.Array([0, 1, -1])
    identity        = af.Array([0, 1, 0] )
    
    """ Number of grid points in the field's domain"""

    (x_number_of_points,  y_number_of_points) = Ez.dims()

    """ number of grid zones from the input fields """

    Nx = x_number_of_points - 2*ghost_cells - 1
    Ny = y_number_of_points - 2*ghost_cells - 1

    """ local variables for storing the input fields """

    Ez_local = Ez.copy()
    Bx_local = Bx.copy()
    By_local = By.copy()

    """Enforcing BC's"""

    Ez_local = periodic_ghost(Ez_local, ghost_cells)

    Bx_local = periodic_ghost(Bx_local, ghost_cells)

    By_local = periodic_ghost(By_local, ghost_cells)

    """ Setting division size and time steps"""

    dx = np.float(length_domain_x / (Nx))
    dy = np.float(length_domain_y / (Ny))

    """ defining variables for convenience """

    dt_by_dx = dt / (dx)
    dt_by_dy = dt / (dy)

    """  Updating the Electric field using the current too """

    Ez_local +=   dt_by_dx * (af.signal.convolve2_separable(identity, backward_column, By_local)) \
              - dt_by_dy * (af.signal.convolve2_separable(backward_row, identity, Bx_local)) \
              - dt*(Jz)

    # dEz/dt = dBy/dx - dBx/dy

    """  Implementing periodic boundary conditions using ghost cells  """

    Ez_local = periodic_ghost(Ez_local, ghost_cells)

    """  Updating the Magnetic fields   """
    
    Bx_local += -dt_by_dy*(af.signal.convolve2_separable(forward_row, identity, Ez_local))

    # dBx/dt = -dEz/dy

    By_local += dt_by_dx*(af.signal.convolve2_separable(identity, forward_column, Ez_local))

    # dBy/dt = +dEz/dx

    """  Implementing periodic boundary conditions using ghost cells  """

    Bx_local = periodic_ghost(Bx_local, ghost_cells)

    By_local = periodic_ghost(By_local, ghost_cells)

    af.eval(Ez_local, Bx_local, By_local)

    return Ez_local, Bx_local, By_local



"""-------------------------------------------------End--of--Mode--1-------------------------------------------------"""


"""-------------------------------------------------Start--of--Mode-2------------------------------------------------"""

""" Equations for mode 2 fdtd (variation along x and y)"""

# dBz/dt = - ( dEy/dx - dEx/dy )
# dEx/dt = + dBz/dy
# dEy/dt = - dBz/dx
# div_B = dBz/dz


def mode2_fdtd( Bz, Ex, Ey, length_domain_x, length_domain_y, ghost_cells, Jx, Jy, Jz, dt,):

    
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

    dx = np.float(length_domain_x / (Nx))
    dy = np.float(length_domain_y / (Ny))

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

    Bz_local += - dt_by_dx * (af.signal.convolve2_separable(identity, forward_column, Ey_local)) \
              + dt_by_dy * (af.signal.convolve2_separable(forward_row, identity, Ex_local))

    # dBz/dt = - ( dEy/dx - dEx/dy )

    #Implementing periodic boundary conditions using ghost cells

    Bz_local = periodic_ghost(Bz_local, ghost_cells)

    af.eval(Bz_local, Ex_local, Ey_local)

    return Bz_local, Ex_local, Ey_local

"""-------------------------------------------------End--of--Mode--2-------------------------------------------------"""

def fdtd(Ex, Ey, Ez, Bx, By, Bz, length_domain_x, length_domain_y, ghost_cells, Jx, Jy, Jz, dt):

    # Decoupling the fields to solve for them individually

    Ez_updated, Bx_updated, By_updated = mode1_fdtd(Ez, Bx, By, length_domain_x, length_domain_y, ghost_cells, Jx, Jy, Jz, dt)

    Bz_updated, Ex_updated, Ey_updated = mode2_fdtd(Bz, Ex, Ey, length_domain_x, length_domain_y, ghost_cells, Jx, Jy, Jz, dt)

    # combining the the results from both modes

    af.eval(Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated)

    return Ex_updated, Ey_updated, Ez_updated, Bx_updated, By_updated, Bz_updated    