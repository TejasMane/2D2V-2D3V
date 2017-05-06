import numpy as np


def set_up_perturbation(positions_x,\
                        positions_y,\
                        number_particles,\
                        N_divisions_x,\
                        N_divisions_y,\
                        amplitude,\
                        k_x,\
                        k_y,\
                        length_domain_x,\
                        length_domain_y,\
                        dx,\
                        dy
                       ):
    '''
    function set_up_perturbation(   positions_x,\
                                    positions_y,\
                                    number_particles,\
                                    N_divisions_x,\
                                    N_divisions_y,\
                                    amplitude,\
                                    k_x,\
                                    k_y,\
                                    length_domain_x,\
                                    length_domain_y,\
                                    dx,\
                                    dy
                                )
    -----------------------------------------------------------------------  
    Input variables: positions_x, number_particles, N_divisions, amplitude, k,length_domain_x

        positions_x, positions_y: An one dimensional array of size equal to number of particles 
        taken in the PIC code. It contains the positions of particles in x and y directions.

        number_particles: The number of electrons /macro particles

        N_divisions_x, N_divisions_y: The number of divisions considered for placing the macro particles
        in x and y directions respectively

        amplitude: This is the amplitude of the density perturbation

        k_x, k_y: The is the wave number of the cosine density pertubation in x and y

        length_domain_x, length_domain_y: This is the length of the domain in x and y directions

        dx, dy: This is the distance between any two consecutive grid nodes of the position grid 
        in x and y directions respectively

    -----------------------------------------------------------------------      
    returns: positions_x, positions_y
        This function returns arrays positions_x and positions_y such that there is a cosine density perturbation 
        of the given amplitude

    '''
    # There might be a few particles left out during execution of function and the statement
    # below randomizes those positions
    positions_x = length_domain_x * np.random.rand(number_particles)
    positions_y = length_domain_y * np.random.rand(number_particles)
    
    particles_till_x_i = 0

    # Looping over grid cells in the domain
    for j in range(N_divisions_y):
        for i in range(N_divisions_x):

            # Average number of particles present between two consecutive grid nodes
            average_particles_x_i_to_i_plus_one = (number_particles/\
                                                   ((length_domain_x * length_domain_y)/(dx * dy))\
                                                  )

            # Amplitude in the current grid cell used to compute number of particles in the
            # current grid cell
            temp_amplitude = amplitude * np.cos((k_x * (i + 0.5) * dx / length_domain_x) + \
                                                (k_y * (j + 0.5) * dy / length_domain_y))

            # Computing number of particles in the current grid cell
            number_particles_x_i_to_i_plus_one = int(average_particles_x_i_to_i_plus_one \
                                                     * (1 + temp_amplitude)\
                                                    )

            # Assigining these number of particles their respective positions in the current grid cell
            positions_x[particles_till_x_i\
                        :particles_till_x_i\
                        + number_particles_x_i_to_i_plus_one \
                       ] \
                                = i * dx \
                                  + dx * np.random.rand(number_particles_x_i_to_i_plus_one)

            positions_y[particles_till_x_i\
                        :particles_till_x_i\
                        + number_particles_x_i_to_i_plus_one \
                       ] \
                                = j * dy \
                                  + dy * np.random.rand(number_particles_x_i_to_i_plus_one)                    
            # keeping track of the number of particles that have been assigned positions
            particles_till_x_i += number_particles_x_i_to_i_plus_one

    return positions_x, positions_y