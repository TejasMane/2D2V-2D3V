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






# Umeda needs x(n), and v(n+0.5dt) for implementation
def Umeda_b1_deposition( charge_electron, positions_x, positions_y, velocity_x, velocity_y,\
                            x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):

    '''
    A modified Umeda's scheme was implemented to handle a pure one dimensional case.

    function Umeda_b1_deposition( charge, x, velocity_x,\
                                  x_grid, ghost_cells, length_domain_x, dt\
                                )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in x direction.
        
        positions_y:  An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles in y direction.

        velocity_x: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the velocities of particles in y direction.
        
        velocity_y: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid in x direction chosen in the PIC simulation
        
        y_grid: This is an array denoting the position grid in y direction chosen in the PIC simulation

        ghost_cells: This is the number of ghost cells used in the simulation domain.

        length_domain_x: This is the length of the domain in x direction

        dt: this is the dt/time step chosen in the simulation
    -----------------------------------------------------------------------
    returns: Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
           Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices

        Jx_x_indices: This returns the x indices (columns) of the array where the respective currents stored in
        Jx_values_at_these_indices have to be deposited
      
        Jx_y_indices: This returns the y indices (rows) of the array where the respective currents stored in
        Jx_values_at_these_indices have to be deposited      

        Jx_values_at_these_indices: This is an array containing the currents to be deposited.
        
        Jy_x_indices, Jy_y_indices and Jy_values_at_these_indices are similar to Jx_x_indices, 
        Jx_y_indices and Jx_values_at_these_indices for Jy

    For further details on the scheme refer to Umeda's paper provided in the sagemath folder as the
    naming conventions used in the function use the papers naming convention.(x_1, x_2, x_r, F_x, )

    '''


    nx = (x_grid.elements() - 1 - 2 * ghost_cells )  # number of zones
    ny = (y_grid.elements() - 1 - 2 * ghost_cells )  # number of zones

    dx = length_domain_x/nx
    dy = length_domain_y/ny

    # Start location x_1, y_1 at t = n * dt
    # Start location x_2, y_2 at t = (n+1) * dt
    
    x_1 = (positions_x).as_type(af.Dtype.f64)
    x_2 = (positions_x + (velocity_x * dt)).as_type(af.Dtype.f64)

    y_1 = (positions_y).as_type(af.Dtype.f64)
    y_2 = (positions_y + (velocity_y * dt)).as_type(af.Dtype.f64)

    # Calculation i_1 and i_2, indices of left corners of cells containing the particles
    # at x_1 and x_2 respectively and j_1 and j_2: indices of bottoms of cells containing the particles
    # at y_1 and y_2 respectively
    
    i_1 = af.arith.floor( ((af.abs( x_1 - af.sum(x_grid[0])))/dx) - ghost_cells)
    j_1 = af.arith.floor( ((af.abs( y_1 - af.sum(y_grid[0])))/dy) - ghost_cells)

    i_2 = af.arith.floor( ((af.abs( x_2 - af.sum(x_grid[0])))/dx) - ghost_cells)
    j_2 = af.arith.floor( ((af.abs( y_2 - af.sum(y_grid[0])))/dy) - ghost_cells)

    i_dx = dx * af.join(1, i_1, i_2)
    j_dy = dy * af.join(1, j_1, j_2)

    i_dx_x_avg = af.join(1, af.max(i_dx,1), ((x_1+x_2)/2))
    j_dy_y_avg = af.join(1, af.max(j_dy,1), ((y_1+y_2)/2))

    x_r_term_1 = dx + af.min(i_dx, 1)
    x_r_term_2 = af.max(i_dx_x_avg, 1)

    y_r_term_1 = dy + af.min(j_dy, 1)
    y_r_term_2 = af.max(j_dy_y_avg, 1)

    x_r_combined_term = af.join(1, x_r_term_1, x_r_term_2)
    y_r_combined_term = af.join(1, y_r_term_1, y_r_term_2)

    # Computing the relay point (x_r, y_r)
    
    
    x_r = af.min(x_r_combined_term, 1)
    y_r = af.min(y_r_combined_term, 1)
    
    # Calculating the fluxes and the weights

    F_x_1 = charge_electron * (x_r - x_1)/dt
    F_x_2 = charge_electron * (x_2 - x_r)/dt

    F_y_1 = charge_electron * (y_r - y_1)/dt
    F_y_2 = charge_electron * (y_2 - y_r)/dt

    W_x_1 = (x_1 + x_r)/(2 * dx) - i_1
    W_x_2 = (x_2 + x_r)/(2 * dx) - i_2

    W_y_1 = (y_1 + y_r)/(2 * dy) - j_1
    W_y_2 = (y_2 + y_r)/(2 * dy) - j_2

    # computing the charge densities at the grid nodes using the 
    # fluxes and the weights
    
    J_x_1_1 = (1/(dx * dy)) * (F_x_1 * (1 - W_y_1))
    J_x_1_2 = (1/(dx * dy)) * (F_x_1 * (W_y_1))

    J_x_2_1 = (1/(dx * dy)) * (F_x_2 * (1 - W_y_2))
    J_x_2_2 = (1/(dx * dy)) * (F_x_2 * (W_y_2))

    J_y_1_1 = (1/(dx * dy)) * (F_y_1 * (1 - W_x_1))
    J_y_1_2 = (1/(dx * dy)) * (F_y_1 * (W_x_1))

    J_y_2_1 = (1/(dx * dy)) * (F_y_2 * (1 - W_x_2))
    J_y_2_2 = (1/(dx * dy)) * (F_y_2 * (W_x_2))

    # concatenating the x, y indices for Jx 
    
    Jx_x_indices = af.join(0,\
                           i_1 + ghost_cells,\
                           i_1 + ghost_cells,\
                           i_2 + ghost_cells,\
                           i_2 + ghost_cells\
                          )

    Jx_y_indices = af.join(0,\
                           j_1 + ghost_cells,\
                           (j_1 + 1 + ghost_cells),\
                            j_2 + ghost_cells,\
                           (j_2 + 1 + ghost_cells)\
                          )
    
    # concatenating the currents at x, y indices for Jx 
    
    Jx_values_at_these_indices = af.join(0,\
                                         J_x_1_1,\
                                         J_x_1_2,\
                                         J_x_2_1,\
                                         J_x_2_2\
                                        )

    # concatenating the x, y indices for Jy 
    
    Jy_x_indices = af.join(0,\
                           i_1 + ghost_cells,\
                           (i_1 + 1 + ghost_cells),\
                            i_2 + ghost_cells,\
                           (i_2 + 1 + ghost_cells)\
                          )

    Jy_y_indices = af.join(0,\
                           j_1 + ghost_cells,\
                           j_1 + ghost_cells,\
                           j_2 + ghost_cells,\
                           j_2 + ghost_cells\
                          )

    # concatenating the currents at x, y indices for Jx 
    
    Jy_values_at_these_indices = af.join(0,\
                                         J_y_1_1,\
                                         J_y_1_2,\
                                         J_y_2_1,\
                                         J_y_2_2\
                                        )

    af.eval(Jx_x_indices, Jx_y_indices, Jy_x_indices, Jy_y_indices)

    af.eval(Jx_values_at_these_indices, Jy_values_at_these_indices)

    return Jx_x_indices, Jx_y_indices, Jx_values_at_these_indices,\
           Jy_x_indices, Jy_y_indices, Jy_values_at_these_indices





# b1 depositor Anti Clockwise d1 to d4 with d1 being the bottom left corner
# and d4 being the bottom right corner
def current_b1_depositor(charge, positions_x, positions_y, velocity_z,\
                         x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y\
                        ):
    
    '''
    function current_b1_depositor(charge, positions_x, positions_y, velocity_z,\
                                  x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y\
                                 )
    -----------------------------------------------------------------------
    Input variables: charge, positions_x, positions_y, velocity_z, x_grid, y_grid, ghost_cells,
     length_domain_x, length_domain_y

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x, positions_y: An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        velocity_z(t = (n+1/2)*dt): An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid, y_grid: These are arrays denoting the position grid chosen in the PIC simulation.

        ghost_cells: This is the number of ghost cells used in the simulation domain..

        length_domain_x, length_domain_y: This is the length of the domain in x and y directions.

    -----------------------------------------------------------------------
    returns: Jz_Yee

        Jz_Yee: This returns the array Jz on their respective Yee yee lattice using direct
        current deposition.


    '''
    
    number_of_electrons = positions_x.elements()
    
    
    x_current_zone = af.data.constant(0, 4 * number_of_electrons, dtype=af.Dtype.u32)
    y_current_zone = af.data.constant(0, 4 * number_of_electrons, dtype=af.Dtype.u32)

    nx = ((x_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones
    ny = ((y_grid.elements()) - 1 - 2 * ghost_cells )  # number of zones

    dx = length_domain_x/nx
    dy = length_domain_y/ny
    
    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(positions_y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

    x_zone_plus = x_zone + 1
    y_zone_plus = y_zone + 1

    dy_by_delta_y = (1/dy) * (positions_y-y_grid[y_zone])
    dy_by_delta_y_complement = 1 - dy_by_delta_y

    dx_by_delta_x = (1/dx) * (positions_x-x_grid[x_zone])
    dx_by_delta_x_complement = 1 - dx_by_delta_x

    weight_corner1 = dy_by_delta_y_complement * dx_by_delta_x_complement
    weight_corner2 = dy_by_delta_y * dx_by_delta_x_complement
    weight_corner3 = dy_by_delta_y * dx_by_delta_x
    weight_corner4 = dy_by_delta_y_complement * dx_by_delta_x

    current_by_dxdy = ((charge/(dx*dy))*velocity_z).as_type(af.Dtype.f64)
    
    # Order of corners is available on the main thesis document
    # order -----bottom right --->bottom left---->top left-----> top right
    
    corner1_currents   = weight_corner1 * current_by_dxdy
    corner2_currents   = weight_corner2 * current_by_dxdy
    corner3_currents   = weight_corner3 * current_by_dxdy
    corner4_currents   = weight_corner4 * current_by_dxdy

    # Concatenating all the currents into one array
    
    all_corners_weighted_current = af.join(0,corner1_currents, corner2_currents,\
                                           corner3_currents, corner4_currents\
                                          )

    # Concatenating all the respective x indices into one array

    x_current_zone[0 * number_of_electrons : 1 * number_of_electrons] = x_zone
    x_current_zone[1 * number_of_electrons : 2 * number_of_electrons] = x_zone
    x_current_zone[2 * number_of_electrons : 3 * number_of_electrons] = x_zone_plus
    x_current_zone[3 * number_of_electrons : 4 * number_of_electrons] = x_zone_plus

    # Concatenating all the respective y indices into one array

    y_current_zone[0 * number_of_electrons : 1 * number_of_electrons] = y_zone
    y_current_zone[1 * number_of_electrons : 2 * number_of_electrons] = y_zone_plus
    y_current_zone[2 * number_of_electrons : 3 * number_of_electrons] = y_zone_plus
    y_current_zone[3 * number_of_electrons : 4 * number_of_electrons] = y_zone

    af.eval(x_current_zone, y_current_zone)
    af.eval(all_corners_weighted_current)

    return x_current_zone, y_current_zone, all_corners_weighted_current





def Umeda_2003(    charge_electron,\
                   number_of_electrons,\
                   positions_x ,positions_y,\
                   velocity_x, velocitiy_y, velocity_z,\
                   x_grid, y_grid,\
                   ghost_cells,\
                   length_domain_x, length_domain_y,\
                   dx, dy,\
                   dt\
              ):

    '''
    function Umeda_2003(   charge_electron,\
                           number_of_electrons,\
                           positions_x ,positions_y,\
                           velocity_x, velocitiy_y, velocity_z,\
                           x_grid, y_grid,\
                           ghost_cells,\
                           length_domain_x, length_domain_y,\
                           dx, dy,\
                           dt\
                      )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x, positions_y(t = n*dt): An one dimensional array of size equal to number of particles taken in the PIC code.
        It contains the positions of particles.

        velocity_x, velocity_y, velocity_z(t = (n+1/2)*dt): An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid, y_grid: Arrays denoting the position grids chosen in the PIC simulation.

        ghost_cells: This is the number of ghost cells used in the simulation domain..

        length_domain_x, length_domain_y: This is the length of the domain in x and y directions.

        dx, dy: distance between consecutive nodes.
        
        dt: this is the dt/time step chosen in the simulation.
    -----------------------------------------------------------------------
    returns: Jx_Yee, Jy_Yee, Jz_Yee

        Jx_Yee, Jy_Yee, Jz_Yee: This returns the array Jx, Jy and Jz on
        their respective Yee yee lattice.


    '''

    elements = x_grid.elements() * y_grid.elements()

    Jx_x_indices, Jx_y_indices,\
    Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = Umeda_b1_deposition(charge_electron,\
                                                     positions_x, positions_y,\
                                                     velocity_x, velocitiy_y,\
                                                     x_grid, y_grid,\
                                                     ghost_cells,\
                                                     length_domain_x, length_domain_y,\
                                                     dt\
                                                   )
    
    # Current deposition using numpy's histogram
    input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices)
    
    Jx_Yee, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jx_values_at_these_indices\
                                     )
    
    Jx_Yee = af.data.moddims(af.to_array(Jx_Yee), y_grid.elements(), x_grid.elements())

    
    # Jy
    input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices)
    
    Jy_Yee, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jy_values_at_these_indices\
                                     )
    
    Jy_Yee = af.data.moddims(af.to_array(Jy_Yee), y_grid.elements(), x_grid.elements())

    
    # Jz
    
    Jz_x_indices, Jz_y_indices,\
    Jz_values_at_these_indices = current_b1_depositor( charge_electron, positions_x,\
                                                       positions_y, velocity_z,\
                                                       x_grid, y_grid, ghost_cells,\
                                                       length_domain_x, length_domain_y\
                                                     )

    input_indices = (Jz_x_indices*(y_grid.elements()) + Jz_y_indices)
    
    Jz_Yee, temp = np.histogram(input_indices,\
                                      bins=elements,\
                                      range=(0, elements),\
                                      weights=Jz_values_at_these_indices\
                                     )
    
    Jz_Yee = af.data.moddims(af.to_array(Jz_Yee),  y_grid.elements(), x_grid.elements())
    
    
    af.eval(Jx_Yee, Jy_Yee, Jz_Yee)

    return Jx_Yee, Jy_Yee, Jz_Yee               











def current_norm_BC_Jx(Jx_Yee, number_of_electrons, w_p, ghost_cells):
    
    '''
    function current_norm_BC(Jx_Yee, number_of_electrons, w_p)
    -----------------------------------------------------------------------  
    Input variables: Jx_Yee, number_of_electrons, w_p

        Jx_Yee: This is an array containing the currents deposited on Yee lattice.

        number_of_electrons: Number of macroparticles taken in the domain.
        
        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------  
    returns: Jx_norm_centered

        Jx_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''       
    
    
    # Normalizing the currents to be deposited
    A                  = 1/(number_of_electrons * w_p)
    
    Jx_norm_Yee  = A * Jx_Yee
    
    # assigning the current density to the boundary points for periodic boundary conditions
    Jx_norm_Yee[:, ghost_cells]  =   Jx_norm_Yee[:, ghost_cells] \
                                         + Jx_norm_Yee[:, -1 - ghost_cells]
    
    
    Jx_norm_Yee[:, -1 - ghost_cells] = Jx_norm_Yee[:, ghost_cells].copy()
    
    
    
    Jx_norm_Yee[:, -2 - ghost_cells] =   Jx_norm_Yee[:, -2 - ghost_cells] \
                                             + Jx_norm_Yee[:, ghost_cells - 1]
        
        
    Jx_norm_Yee[:, ghost_cells + 1] =   Jx_norm_Yee[:, ghost_cells + 1]\
                                             + Jx_norm_Yee[:, -ghost_cells]
        
    
    
    # assigning the current density to the boundary points in top and bottom rows along y direction
    Jx_norm_Yee[ghost_cells, :] = Jx_norm_Yee[ghost_cells, :] \
                                        + Jx_norm_Yee[-1-ghost_cells, :]
        
    Jx_norm_Yee[-1-ghost_cells, :] = Jx_norm_Yee[ghost_cells, :].copy()
    
    Jx_norm_Yee[ghost_cells + 1, :] = Jx_norm_Yee[ghost_cells + 1, :] +Jx_norm_Yee[-ghost_cells, :]
    Jx_norm_Yee[-2 - ghost_cells, :] =   Jx_norm_Yee[-2 - ghost_cells, :] \
                                             + Jx_norm_Yee[ghost_cells - 1, :]
    
    # Assigning ghost cell values
    Jx_norm_Yee = periodic_ghost(Jx_norm_Yee, ghost_cells)
    
    af.eval(Jx_norm_Yee)
    
    return Jx_norm_Yee



def current_norm_BC_Jy(Jy_Yee, number_of_electrons, w_p, ghost_cells):
    
    '''
    function current_norm_BC(Jy_Yee, number_of_electrons, w_p)
    -----------------------------------------------------------------------  
    Input variables: Jx_Yee, number_of_electrons, w_p

        Jy_Yee: This is an array containing the currents deposited on Yee lattice.

        number_of_electrons: Number of macroparticles taken in the domain.
        
        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------  
    returns: Jy_norm_centered

        Jy_norm_centered: This returns the array Jx on the centered lattice same as the electric field.


    '''   
    
    
    # Normalizing the currents to be deposited
    A                  = 1/(number_of_electrons * w_p)
    
    Jy_norm_Yee  = A * Jy_Yee
    
    
    # assigning the current density to the boundary points for periodic boundary conditions
    Jy_norm_Yee[ghost_cells, :]  =   Jy_norm_Yee[ghost_cells, :] \
                                         + Jy_norm_Yee[-1 - ghost_cells, :]
    
    Jy_norm_Yee[-1 - ghost_cells, :] = Jy_norm_Yee[ghost_cells, :].copy()
    
    
    
    Jy_norm_Yee[-2 - ghost_cells, :] =   Jy_norm_Yee[-2 - ghost_cells, :] \
                                             + Jy_norm_Yee[ghost_cells - 1, :]
        
    
    Jy_norm_Yee[ghost_cells + 1, :] =   Jy_norm_Yee[ghost_cells + 1, :]\
                                             + Jy_norm_Yee[-ghost_cells, :]
    
    # assigning the current density to the boundary points in left and right columns along x direction
    Jy_norm_Yee[:, ghost_cells] = Jy_norm_Yee[:, ghost_cells] + Jy_norm_Yee[:, -1-ghost_cells]
    Jy_norm_Yee[:, -1-ghost_cells] = Jy_norm_Yee[:, ghost_cells].copy()
    
    
    Jy_norm_Yee[:, ghost_cells + 1] = Jy_norm_Yee[:, ghost_cells + 1] +Jy_norm_Yee[:, -ghost_cells]
    Jy_norm_Yee[:, -2 - ghost_cells] =   Jy_norm_Yee[:, -2 - ghost_cells] \
                                             + Jy_norm_Yee[:, ghost_cells - 1]
    
    
    # Assigning ghost cell values
    Jy_norm_Yee = periodic_ghost(Jy_norm_Yee, ghost_cells)
    
    
    af.eval(Jy_norm_Yee)
    
    return Jy_norm_Yee    



def current_norm_BC_Jz(Jz_Yee, number_of_electrons, w_p, ghost_cells):
    
    '''
    function current_norm_BC(Jz_Yee, number_of_electrons, w_p)
    -----------------------------------------------------------------------  
    Input variables: Jz_Yee, number_of_electrons, w_p

        Jz_Yee: This is an array containing the currents deposited on yee lattice.

        number_of_electrons: Number of macroparticles taken in the domain.
        
        w_p: Number of particles comprising the macroparticle.

    -----------------------------------------------------------------------  
    returns: Jz_norm_Yee

        Jz_norm_Yee: This returns the array Jx on the centered lattice same as Ez.


    '''   
    
    
    # Normalizing the currents to be deposited
    A                  = 1/(number_of_electrons * w_p)
    
    Jz_norm_Yee  = A * Jz_Yee
    
    
    # assigning the current density to the boundary points for periodic boundary conditions
    Jz_norm_Yee[ghost_cells, :] = Jz_norm_Yee[ghost_cells, :] + Jz_norm_Yee[-1-ghost_cells, :]
    Jz_norm_Yee[-1-ghost_cells, :] = Jz_norm_Yee[ghost_cells, :].copy()
    
    Jz_norm_Yee[ghost_cells + 1, :] =   Jz_norm_Yee[ghost_cells + 1, :]\
                                         + Jz_norm_Yee[-ghost_cells, :]
    
    Jz_norm_Yee[-2 - ghost_cells, :] =   Jz_norm_Yee[-2 - ghost_cells, :] \
                                         + Jz_norm_Yee[ghost_cells - 1, :]
        
    # assigning the current density to the boundary points in left and right columns along x direction
    Jz_norm_Yee[:, ghost_cells] = Jz_norm_Yee[:, ghost_cells] + Jz_norm_Yee[:, -1-ghost_cells]
    Jz_norm_Yee[:, -1-ghost_cells] = Jz_norm_Yee[:, ghost_cells].copy()
    
    Jz_norm_Yee[:, ghost_cells + 1] = Jz_norm_Yee[:, ghost_cells + 1] +Jz_norm_Yee[:, -ghost_cells]
    Jz_norm_Yee[:, -2 - ghost_cells] =   Jz_norm_Yee[:, -2 - ghost_cells] \
                                             + Jz_norm_Yee[:, ghost_cells - 1]
    
    # Assigning ghost cell values
    Jz_norm_Yee = periodic_ghost(Jz_norm_Yee, ghost_cells)
    
    
    af.eval(Jz_norm_Yee)
    
    return Jz_norm_Yee