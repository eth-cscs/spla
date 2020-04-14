# SPLA - Specialized Parallel Linear Algebra
SPLA provides specialized functions for linear algebra computations, which are inspired by requirements in computational material science codes.

The implementations is based on MPI and optinally utilized OpenMP and GPU acceleration through CUDA or ROCm.
## GEMM
Currently, two general matrix multiplication functions are available, which allow for different matrix distributions as input and output.
These specific configurations cannot be directly expressed with the commonly used p?gemm function of ScaLAPACK.

### Stripe-Stripe-Block
The `gemm_ssb(...)` function computes the following:  
![ethz](docs/images/ssb_formula.png)

The matrices A and B are stored in a "stripe" distribution with variable block length. Matrix C can be in any supported block distribution.
See documentation for details. 


     ------ H     ------
     |    |       |    |
     |    |       |    |
     ------       ------
     |    |       |    |        -------
     |    |       |    |        |  |  |
     ------   *   ------    +   -------
     |    |       |    |        |  |  |
     |    |       |    |        -------
     ------       ------           C
     |    |       |    |
     |    |       |    |
     ------       ------
       A            B



### Stripe-Block-Stripe
The `gemm_sbs(...)` function computes the following:  
![ethz](docs/images/sbs_formula.png)

The matrices A and C are stored in a "stripe" distribution with variable block length. Matrix B can be in any supported block distribution.
See documentation for details. 

     ------                     ------
     |    |                     |    |
     |    |                     |    |
     ------                     ------
     |    |       -------       |    |
     |    |       |  |  |       |    |
     ------   *   -------   +   ------
     |    |       |  |  |       |    |
     |    |       -------       |    |
     ------          B          ------
     |    |                     |    |
     |    |                     |    |
     ------                     ------
       A                          C
## Acknowledgements
The development of SPLA would not be possible without support of the following organizations:

| Logo | Name | URL |
|:----:|:----:|:---:|
|![ethz](docs/images/logo_ethz.png) | Swiss Federal Institute of Technology in Zürich | https://www.ethz.ch/      |
|![cscs](docs/images/logo_cscs.png) | Swiss National Supercomputing Centre            | https://www.cscs.ch/      |
|![pasc](docs/images/logo_max.png)  | MAX (MAterials design at the eXascale) <br> European Centre of Excellence | http://www.max-centre.eu/   |
