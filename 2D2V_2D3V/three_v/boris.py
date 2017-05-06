import arrayfire as af

def periodic_particles(positions_x, positions_y, positions_z, length_domain_x, length_domain_y, length_domain_z):
    '''
    function periodic_particles(positions_x, positions_y, positions_z, length_domain_x, length_domain_y, length_domain_z)
    -----------------------------------------------------------------------
    Input variables: positions_x and length_domain_x

      positions_x, positions_y, positions_z: One dimensional arrays of size equal to number of particles
      taken in the PIC code. It contains the positions of particles in x and y directions
      respectively

      length_domain_x, length_domain_y, length_domain_z: This is the length of the domain in x,y and z directions respectively.
      The domain is assumed to be from x = 0 to x = length_domain_x

    -----------------------------------------------------------------------
    returns: positions_x, positions_y, positions_z
      This function returns the modified positions_x and positions_y such that particle previously
      gone outside the domain through the left boundary enter via the right boundary and vice versa.
      In other words, the function implements periodic boundary conditions for the particles.

    '''

    # Arrayfire implementation
    # Determine indices of particles which have gone outside the domain
    # through right boundary

    outside_domain_right_x       = af.algorithm.where(positions_x >= length_domain_x)
    outside_domain_top_y         = af.algorithm.where(positions_y >= length_domain_y)
    outside_domain_up_z          = af.algorithm.where(positions_z >= length_domain_z)

    # Determine indices of particles which have gone outside the domain
    # through left boundary
    outside_domain_left_x        = af.algorithm.where(positions_x <  0  )
    outside_domain_bottom_y      = af.algorithm.where(positions_y <  0  )
    outside_domain_down_z        = af.algorithm.where(positions_z <  0  )


    if outside_domain_right_x.elements() > 0:

    # Apply periodic boundary conditions

      positions_x[outside_domain_right_x] = positions_x[outside_domain_right_x] - length_domain_x

    if outside_domain_top_y.elements() > 0:

    # Apply periodic boundary conditions

      positions_y[outside_domain_top_y] = positions_y[outside_domain_top_y] - length_domain_y

    if outside_domain_up_z.elements() > 0:

    # Apply periodic boundary conditions

      positions_z[outside_domain_up_z] = positions_z[outside_domain_up_z] - length_domain_z


    if outside_domain_left_x.elements() > 0:

    # Apply periodic boundary conditions

      positions_x[outside_domain_left_x]  = positions_x[outside_domain_left_x] + length_domain_x

    if outside_domain_bottom_y.elements() > 0:

    # Apply periodic boundary conditions

      positions_y[outside_domain_bottom_y]  = positions_y[outside_domain_bottom_y] + length_domain_y

    if outside_domain_down_z.elements() > 0:

    # Apply periodic boundary conditions

      positions_z[outside_domain_down_z]  = positions_z[outside_domain_down_z] + length_domain_z

    af.eval(positions_x, positions_y, positions_z)

    return positions_x, positions_y, positions_z


def Boris(mass_electron, charge_electron, vel_x, vel_y, vel_z, dt, Ex, Ey, Ez, Bx, By, Bz):
    
    '''
    function Boris(mass_electron, charge_electron, vel_x, vel_y, vel_z, dt, Ex, Ey, Ez, Bx, By, Bz)
    -----------------------------------------------------------------------  
    Input variables: mass_electron, charge_electron, vel_x, vel_y, vel_z, dt, Ex, Ey, Ez, Bx, By, Bz

    mass_electron: mass of the macro particle
    
    charge_electron: charge of the macro particle
    
    vel_x, vel_y, vel_z: The input velocities in x, y, z
    
    dt: Time step in the PIC code
    
    Ex, Ey, Ez, Bx, By, Bz: These are the interpolated fields E(x((n+1) * dt)), B(x((n+1) * dt)) used
    to push v((n+1/2) * dt) to v((n+3/2) * dt)
    -----------------------------------------------------------------------  
    returns: vel_x_new, vel_y_new, vel_z_new

        vel_x_new, vel_y_new, vel_z_new: These are the updated velocity arrays from 
        v((n+1/2)*dt) to v((n+3/2)*dt)


    '''       
    
    vel_x_minus = vel_x + (charge_electron * Ex * dt) / (2 * mass_electron)
    vel_y_minus = vel_y + (charge_electron * Ey * dt) / (2 * mass_electron)
    vel_z_minus = vel_z + (charge_electron * Ez * dt) / (2 * mass_electron)

    t_magx    = (charge_electron * Bx * dt) / (2 * mass_electron)
    t_magy    = (charge_electron * By * dt) / (2 * mass_electron)
    t_magz    = (charge_electron * Bz * dt) / (2 * mass_electron)

    vminus_cross_t_x =  (vel_y_minus * t_magz) - (vel_z_minus * t_magy)
    vminus_cross_t_y = -(vel_x_minus * t_magz) + (vel_z_minus * t_magx)
    vminus_cross_t_z =  (vel_x_minus * t_magy) - (vel_y_minus * t_magx)

    vel_dashx = vel_x_minus + vminus_cross_t_x
    vel_dashy = vel_y_minus + vminus_cross_t_y
    vel_dashz = vel_z_minus + vminus_cross_t_z

    t_mag = af.arith.sqrt(t_magx ** 2 + t_magy ** 2 + t_magz ** 2)

    s_x = (2 * t_magx) / (1 + af.arith.abs(t_mag ** 2))
    s_y = (2 * t_magy) / (1 + af.arith.abs(t_mag ** 2))
    s_z = (2 * t_magz) / (1 + af.arith.abs(t_mag ** 2))

    vel_x_plus = vel_x_minus + ((vel_dashy * s_z) - (vel_dashz * s_y))
    vel_y_plus = vel_y_minus - ((vel_dashx * s_z) - (vel_dashz * s_x))
    vel_z_plus = vel_z_minus + ((vel_dashx * s_y) - (vel_dashy * s_x))

    vel_x_new  = vel_x_plus + (charge_electron * Ex * dt) / (2 * mass_electron)
    vel_y_new  = vel_y_plus + (charge_electron * Ey * dt) / (2 * mass_electron)
    vel_z_new  = vel_z_plus + (charge_electron * Ez * dt) / (2 * mass_electron)

    af.eval(vel_x_new, vel_y_new, vel_z_new)

    return (vel_x_new, vel_y_new, vel_z_new)