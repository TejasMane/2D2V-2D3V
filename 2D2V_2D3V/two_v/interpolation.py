import arrayfire as af


def fraction_finder(positions_x, positions_y, x_grid, y_grid, dx, dy):
    '''
    function fraction_finder(positions_x, positions_y, x_grid, y_grid, dx, dy)
    -----------------------------------------------------------------------
    Input variables: positions_x and length_domain_x

        positions_x: An one dimensional array of size equal to number of particles taken in the PIC code. 
        It contains the positions of particles in x direction.
        
        positions_y: An one dimensional array of size equal to number of particles taken in the PIC code. 
        It contains the positions of particles in y direction.
        
        x_grid, y_grid: This is an array denoting the position grid chosen in the PIC simulation in
        x and y directions respectively
        

        dx, dy: This is the distance between any two consecutive grid nodes of the position grid 
        in x and y directions respectively

    -----------------------------------------------------------------------    
    returns: x_frac, y_frac
        This function returns the fractions of grid cells needed to perform the 2D charge deposition 

    '''
    x_frac = (positions_x - af.sum(x_grid[0])) / dx
    y_frac = (positions_y - af.sum(y_grid[0])) / dy

    af.eval(x_frac, y_frac)

    return x_frac, y_frac