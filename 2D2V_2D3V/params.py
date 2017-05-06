import numpy as np

# Weighting factor w_p = number of electrons composing the macro particle
# Doesn't affect the the physics of the system (Coarse graining doesn't affect vlasov maxwell system)
# It should be kept such that the number density in the PIC code corresponds
# to the densities found in real plasmas
w_p             = 1

# Macro Particle parameters
k_boltzmann     = 1
mass_electron   = 1 * w_p
tempertature    = 1
charge_electron = -10 * w_p
charge_ion      = +10 * w_p

# Setting the length of the domain
length_domain_x = 1
length_domain_y = 1
length_domain_z = 1

# Setting the number of ghost cells
ghost_cells  = 1

# Setting number of particle in the domain
number_of_electrons = 400000

# Divisions in x grid
divisions_domain_x = 100
divisions_domain_y = 100

amplitude_perturbed = 0.5
k_x                 = 2 * np.pi
k_y                 = 2 * np.pi