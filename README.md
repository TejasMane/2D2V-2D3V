This repository contains all the code used for the 2D3V Particle in Cell simualations


The particle in cell (PIC) method used to solve the Vlasov-Maxwell system of partial differential equations was implemented. The PIC code was developed using the equations for motion for macro particles, charge and current deposition. Charge conserving current deposition schemes were used to address the violation of the continuity equation. The Landau damping predicted by analytical calculations was compared with the damping observed via the PIC code. Higher order shape factors and alternative methods were employed to minimize the high margin of error resulting from particle noise. The Python code's features include: 
* An particle pusher based on the Boris algorithm.
* A Fast Fourier Transform(FFT) and Successive Over Relaxation(SOR) based Poisson solver.
* A Finite Difference Time Domain(FDTD) algorithm to evolve the electric and magnetic fields in the domain.
* A direct charge deposition scheme with charge conserving current deposition schemes based on the Umeda et.al (2003) and Esirkepov (2001) algorithms.
* Support for higher order B-splines based shape factors and completely vectorized using Arrayfire's high performance libraries capable of running on CPU's, OpenCL and Nvidia CUDA devices.
