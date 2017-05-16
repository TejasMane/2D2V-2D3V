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






def histogram_deposition(current_indices_flat, currents_flat, grid_elements):
    
    '''
    function: histogram_deposition(current_indices_flat, currents_flat, grid)
    
    inputs: current_indices_flat, currents_flat, grid_elements
    
    current_indices_flat, currents_flat: They denote the indices and the currents
    to be deposited on the flattened current vector.
    
    grid_elements: The number of elements present the matrix/vector representing the 
    currents.   
    
    
    '''
    
    
    # setting default indices and current for histogram deposition
    indices_fix = af.data.range(grid_elements + 1, dtype = af.Dtype.s64)
    currents_fix = 0 * af.data.range(grid_elements + 1, dtype = af.Dtype.f64)
    
    
    # Concatenating the indices and currents in a single vector 
    
    combined_indices_flat  = af.join(0, indices_fix, current_indices_flat)
    combined_currents_flat = af.join(0, currents_fix, currents_flat)
    
    
    # Sort by key operation
    indices, currents = af.sort_by_key(combined_indices_flat, combined_currents_flat, dim=0)
    
    # scan by key operation with default binary addition operation which sums up currents 
    # for the respective indices
    
    Histogram_scan = af.scan_by_key(indices, currents)
    
    # diff1 operation to determine the uniques indices in the current
    diff1_op = af.diff1(indices, dim = 0)
    
    
    # Determining the uniques indices for current deposition
    indices_unique = af.where(diff1_op > 0)
    
    # Derming the current vector
    
    J_flat = Histogram_scan[indices_unique]
    
    af.eval(J_flat)
    
    return J_flat






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





def Umeda_2003(    charge_electron,\
                   number_of_electrons,\
                   positions_x ,positions_y,\
                   velocities_x, velocities_y,\
                   x_grid, y_grid,\
                   ghost_cells,\
                   length_domain_x, length_domain_y,\
                   dx, dy,\
                   dt\
              ):

    '''
    function Umeda_b1_deposition( charge, x, velocity_x,\
                                  x_grid, ghost_cells, length_domain_x, dt\
                                )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge: This is an array containing the charges deposited at the density grid nodes.

        positions_x(t = n*dt): An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the positions of particles.

        velocity_x(t = (n+1/2)*dt): An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid: This is an array denoting the position grid chosen in the PIC simulation.

        ghost_cells: This is the number of ghost cells used in the simulation domain..

        length_domain_x: This is the length of the domain in x direction.

        dt: this is the dt/time step chosen in the simulation.
    -----------------------------------------------------------------------
    returns: Jx_Yee, Jy_Yee

        Jx_Yee, Jy_Yee: This returns the array Jx and Jy on their respective Yee 
        yee lattice.


    '''

    elements = x_grid.elements() * y_grid.elements()

    Jx_x_indices, Jx_y_indices,\
    Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = Umeda_b1_deposition(charge_electron,\
                                                     positions_x, positions_y,\
                                                     velocities_x, velocities_y,\
                                                     x_grid, y_grid,\
                                                     ghost_cells,\
                                                     length_domain_x, length_domain_y,\
                                                     dt\
                                                   )
    
    # Current deposition using numpy's histogram
    input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices).as_type(af.Dtype.s64)
        
    # Computing Jx_Yee
    
    Jx_Yee = histogram_deposition(input_indices, Jx_values_at_these_indices, elements)
    
    Jx_Yee = af.data.moddims(af.to_array(Jx_Yee), y_grid.elements(), x_grid.elements())
    
    # Computing Jy_Yee
    
    input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices).as_type(af.Dtype.s64)
    
    Jy_Yee = histogram_deposition(input_indices, Jy_values_at_these_indices, elements)
    
    Jy_Yee = af.data.moddims(af.to_array(Jy_Yee), y_grid.elements(), x_grid.elements())

    af.eval(Jx_Yee, Jy_Yee)

    return Jx_Yee, Jy_Yee




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



def indices_and_currents_TSC_2D( charge_electron, positions_x, positions_y, velocity_x, velocity_y,\
                            x_grid, y_grid, ghost_cells, length_domain_x, length_domain_y, dt  ):
    """
    
    function indices_and_currents_TSC_2D( charge_electron, positions_x,\
                                          positions_y, velocity_x, velocity_y,\
                                          x_grid, y_grid, ghost_cells,\
                                          length_domain_x, length_domain_y, dt\
                                        )
    return the x and y indices for Jx and Jy and respective currents associated with these indices 
    
    """
    
    
    positions_x_new     = positions_x + velocity_x * dt
    positions_y_new     = positions_y + velocity_y * dt

    base_indices_x = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)
    base_indices_y = af.data.constant(0, positions_x.elements(), dtype=af.Dtype.u32)

    dx = af.sum(x_grid[1] - x_grid[0])
    dy = af.sum(y_grid[1] - y_grid[0])


    # Computing S0_x and S0_y
    ###########################################################################################
    
    # Determining the grid cells containing the respective particles
    
    x_zone = (((af.abs(positions_x - af.sum(x_grid[0])))/dx).as_type(af.Dtype.u32))
    y_zone = (((af.abs(positions_y - af.sum(y_grid[0])))/dy).as_type(af.Dtype.u32))

    
    # Determing the indices of the closest grid node in x direction

    temp = af.where(af.abs(positions_x-x_grid[x_zone]) < \
                    af.abs(positions_x-x_grid[x_zone + 1])\
                   )

    if(temp.elements()>0):
        base_indices_x[temp] = x_zone[temp]

    temp = af.where(af.abs(positions_x - x_grid[x_zone]) >= \
                    af.abs(positions_x-x_grid[x_zone + 1])\
                   )

    if(temp.elements()>0):
        base_indices_x[temp] = (x_zone[temp] + 1).as_type(af.Dtype.u32)    


    # Determing the indices of the closest grid node in y direction

    temp = af.where(af.abs(positions_y-y_grid[y_zone]) < \
                    af.abs(positions_y-y_grid[y_zone + 1])\
                   )

    if(temp.elements()>0):
        base_indices_y[temp] = y_zone[temp]

    temp = af.where(af.abs(positions_y - y_grid[y_zone])>=af.abs(positions_y-x_grid[y_zone + 1]))

    if(temp.elements()>0):
        base_indices_y[temp] = (y_zone[temp] + 1).as_type(af.Dtype.u32)  

    # Concatenating the index list for near by grid nodes in x direction
    # TSC affect 5 nearest grid nodes around in 1 Dimensions

    base_indices_minus_two = (base_indices_x - 2).as_type(af.Dtype.u32)    
    base_indices_minus     = (base_indices_x - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_x + 1).as_type(af.Dtype.u32)    
    base_indices_plus_two  = (base_indices_x + 2).as_type(af.Dtype.u32)    



    index_list_x = af.join( 1,\
                             af.join(1, base_indices_minus_two, base_indices_minus, base_indices_x),\
                             af.join(1, base_indices_plus, base_indices_plus_two),\
                          )



    # Concatenating the index list for near by grid nodes in y direction
    # TSC affect 5 nearest grid nodes around in 1 Dimensions
    
    base_indices_minus_two = (base_indices_y - 2).as_type(af.Dtype.u32)    
    base_indices_minus     = (base_indices_y - 1).as_type(af.Dtype.u32)    
    base_indices_plus      = (base_indices_y + 1).as_type(af.Dtype.u32)    
    base_indices_plus_two  = (base_indices_y + 2).as_type(af.Dtype.u32)     


    index_list_y = af.join( 1,\
                             af.join(1, base_indices_minus_two, base_indices_minus, base_indices_y),\
                             af.join(1, base_indices_plus, base_indices_plus_two),\
                          )

    # Concatenating the positions_x for determining weights for near by grid nodes in y direction
    # TSC affect 5 nearest grid nodes around in 1 Dimensions

    positions_x_5x        = af.join( 0,\
                                     af.join(0, positions_x, positions_x, positions_x),\
                                     af.join(0, positions_x, positions_x),\
                                   )

    positions_y_5x        = af.join( 0,\
                                     af.join(0, positions_y, positions_y, positions_y),\
                                     af.join(0, positions_y, positions_y),\
                                   )




    # Determining S0 for positions at t = n * dt


    distance_nodes_x = x_grid[af.flat(index_list_x)]

    distance_nodes_y = y_grid[af.flat(index_list_y)]


    W_x = 0 * distance_nodes_x.copy()
    W_y = 0 * distance_nodes_y.copy()


    # Determining weights in x direction

    temp = af.where(af.abs(distance_nodes_x - positions_x_5x) < (0.5*dx) )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_5x[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_x - positions_x_5x) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_x - positions_x_5x) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] - positions_x_5x[temp])/dx))**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_5x) < (0.5*dy) )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] - positions_y_5x[temp])/dy)**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_5x) >= (0.5*dy) )\
                     * (af.abs(distance_nodes_y - positions_y_5x) < (1.5 * dy) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] - positions_y_5x[temp])/dy))**2

    # Restructering W_x and W_y for visualization and ease of understanding

    W_x = af.data.moddims(W_x, positions_x.elements(), 5)
    W_y = af.data.moddims(W_y, positions_y.elements(), 5)

    # Tiling the S0_x and S0_y for the 25 indices around the particle
    
    S0_x = af.tile(W_x, 1, 1, 5)
    S0_y = af.tile(W_y, 1, 1, 5)


    S0_y = af.reorder(S0_y, 0, 2, 1)



    #Computing S1_x and S1_y
    ###########################################################################################

    positions_x_5x_new    = af.join( 0,\
                                     af.join(0, positions_x_new, positions_x_new, positions_x_new),\
                                     af.join(0, positions_x_new, positions_x_new),\
                                   )

    positions_y_5x_new    = af.join( 0,\
                                     af.join(0, positions_y_new, positions_y_new, positions_y_new),\
                                     af.join(0, positions_y_new, positions_y_new),\
                                   )




    # Determining S0 for positions at t = n * dt

    W_x = 0 * distance_nodes_x.copy()
    W_y = 0 * distance_nodes_y.copy()


    # Determining weights in x direction

    temp = af.where(af.abs(distance_nodes_x - positions_x_5x_new) < (0.5*dx) )

    if(temp.elements()>0):
        W_x[temp] = 0.75 - (af.abs(distance_nodes_x[temp] - positions_x_5x_new[temp])/dx)**2

    temp = af.where((af.abs(distance_nodes_x - positions_x_5x_new) >= (0.5*dx) )\
                     * (af.abs(distance_nodes_x - positions_x_5x_new) < (1.5 * dx) )\
                   )

    if(temp.elements()>0):
        W_x[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_x[temp] \
                                  - positions_x_5x_new[temp])/dx\
                                 )\
                          )**2



    # Determining weights in y direction

    temp = af.where(af.abs(distance_nodes_y - positions_y_5x_new) < (0.5*dy) )

    if(temp.elements()>0):
        W_y[temp] = 0.75 - (af.abs(distance_nodes_y[temp] \
                                   - positions_y_5x_new[temp]\
                                  )/dy\
                           )**2

    temp = af.where((af.abs(distance_nodes_y - positions_y_5x_new) >= (0.5*dy) )\
                     * (af.abs(distance_nodes_y - positions_y_5x_new) < (1.5 * dy) )\
                   )

    if(temp.elements()>0):
        W_y[temp] = 0.5 * (1.5 - (af.abs(distance_nodes_y[temp] \
                                         - positions_y_5x_new[temp])/dy\
                                 )\
                          )**2

    # Restructering W_x and W_y for visualization and ease of understanding

    W_x = af.data.moddims(W_x, positions_x.elements(), 5)
    W_y = af.data.moddims(W_y, positions_x.elements(), 5)

    # Tiling the S0_x and S0_y for the 25 indices around the particle    
    
    S1_x = af.tile(W_x, 1, 1, 5)
    S1_y = af.tile(W_y, 1, 1, 5)

    S1_y = af.reorder(S1_y, 0, 2, 1)


    ###########################################################################################

    # Determining the final weight matrix for currents in 3D matrix form factor


    W_x = (S1_x - S0_x) * (S0_y + (0.5 *(S1_y - S0_y)) )


    W_y = (S1_y - S0_y) * (S0_x + (0.5 *(S1_x - S0_x)) )


    ###########################################################################################


    # Assigning Jx and Jy according to Esirkepov's scheme

    Jx = af.data.constant(0, positions_x.elements(), 5, 5, dtype = af.Dtype.f64)
    Jy = af.data.constant(0, positions_x.elements(), 5, 5, dtype = af.Dtype.f64)


    Jx[:, 0, :] = -1 * charge_electron * (dx/dt) * W_x[:, 0, :].copy()
    Jx[:, 1, :] = Jx[:, 0, :] + -1 * charge_electron * (dx/dt) * W_x[:, 1, :].copy()
    Jx[:, 2, :] = Jx[:, 1, :] + -1 * charge_electron * (dx/dt) * W_x[:, 2, :].copy()
    Jx[:, 3, :] = Jx[:, 2, :] + -1 * charge_electron * (dx/dt) * W_x[:, 3, :].copy()
    Jx[:, 4, :] = Jx[:, 3, :] + -1 * charge_electron * (dx/dt) * W_x[:, 4, :].copy()
    
    # Computing current density using currents
    
    Jx = (1/(dx * dy)) * Jx


    Jy[:, :, 0] = -1 * charge_electron * (dy/dt) * W_y[:, :, 0].copy()
    Jy[:, :, 1] = Jy[:, :, 0] + -1 * charge_electron * (dy/dt) * W_y[:, :, 1].copy()
    Jy[:, :, 2] = Jy[:, :, 1] + -1 * charge_electron * (dy/dt) * W_y[:, :, 2].copy()
    Jy[:, :, 3] = Jy[:, :, 2] + -1 * charge_electron * (dy/dt) * W_y[:, :, 3].copy()
    Jy[:, :, 4] = Jy[:, :, 3] + -1 * charge_electron * (dy/dt) * W_y[:, :, 4].copy()
    
    # Computing current density using currents

    Jy = (1/(dx * dy)) * Jy

    # Preparing the final index and current vectors
    ###########################################################################################
    
    
    # Determining the x indices for charge deposition
    index_list_x_Jx = af.flat(af.tile(index_list_x, 1, 1, 5))

    # Determining the y indices for charge deposition
    y_current_zone = af.tile(index_list_y, 1, 1, 5)
    index_list_y_Jx = af.flat(af.reorder(y_current_zone, 0, 2, 1))


    currents_Jx = af.flat(Jx)

    # Determining the x indices for charge deposition
    index_list_x_Jy = af.flat(af.tile(index_list_x, 1, 1, 5))

    # Determining the y indices for charge deposition
    y_current_zone = af.tile(index_list_y, 1, 1, 5)
    index_list_y_Jy = af.flat(af.reorder(y_current_zone, 0, 2, 1))
    
    # Flattenning the Currents array
    currents_Jy = af.flat(Jy)

    af.eval(index_list_x_Jx, index_list_y_Jx)
    af.eval(index_list_x_Jy, index_list_y_Jy)
    af.eval(currents_Jx, currents_Jy)


    return index_list_x_Jx, index_list_y_Jx, currents_Jx,\
           index_list_x_Jy, index_list_y_Jy, currents_Jy
        
        
        
def Esirkepov_2D(  charge_electron,\
                   number_of_electrons,\
                   positions_x ,positions_y,\
                   velocities_x, velocities_y,\
                   x_grid, y_grid,\
                   ghost_cells,\
                   length_domain_x, length_domain_y,\
                   dx, dy,\
                   dt\
               ):
    
    '''
    function Esirkepov_2D( charge_electron,\
                           number_of_electrons,\
                           positions_x ,positions_y,\
                           velocities_x, velocities_y,\
                           x_grid, y_grid,\
                           ghost_cells,\
                           length_domain_x, length_domain_y,\
                           dx, dy,\
                           dt\
                       )
    -----------------------------------------------------------------------
    Input variables: charge, x, velocity_required_x, x_grid, ghost_cells, length_domain_x, dt

        charge_electron: This is an array containing the charges deposited at the density grid nodes.

        positions_x(t = n*dt), positions_y: An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the positions of particles in x and y directions

        velocity_x(t = (n+1/2)*dt), velocities_y: An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the velocities of particles in x direction.

        x_grid, y_grid: This is an array denoting the position grid chosen in the PIC simulation 
        in x and y directions.

        ghost_cells: This is the number of ghost cells used in the simulation domain.

        length_domain_x, length_domain_x: This is the length of the domain in x and y directions.

        dt: this is the dt/time step chosen in the simulation.
    -----------------------------------------------------------------------
    returns: Jx_Yee, Jy_Yee

        Jx_Yee, Jy_Yee: This returns the array Jx and Jy on their respective Yee 
        yee lattice.


    '''
    
    
    elements = x_grid.elements() * y_grid.elements()

    Jx_x_indices, Jx_y_indices,\
    Jx_values_at_these_indices,\
    Jy_x_indices, Jy_y_indices,\
    Jy_values_at_these_indices = indices_and_currents_TSC_2D(charge_electron,\
                                                             positions_x, positions_y,\
                                                             velocities_x, velocities_y,\
                                                             x_grid, y_grid,\
                                                             ghost_cells,\
                                                             length_domain_x, length_domain_y,\
                                                             dt\
                                                           )
    
    # Current deposition using numpy's histogram
    input_indices = (Jx_x_indices*(y_grid.elements()) + Jx_y_indices).as_type(af.Dtype.s64)
    
    # Computing Jx_Yee
    
    Jx_Yee = histogram_deposition(input_indices, Jx_values_at_these_indices, elements)
    
    Jx_Yee = af.data.moddims(af.to_array(Jx_Yee), y_grid.elements(), x_grid.elements())
    
    # Computing Jy_Yee
    
    input_indices = (Jy_x_indices*(y_grid.elements()) + Jy_y_indices).as_type(af.Dtype.s64)
    
    Jy_Yee = histogram_deposition(input_indices, Jy_values_at_these_indices, elements)
    
    Jy_Yee = af.data.moddims(af.to_array(Jy_Yee), y_grid.elements(), x_grid.elements())

    af.eval(Jx_Yee, Jy_Yee)

    return Jx_Yee, Jy_Yee
