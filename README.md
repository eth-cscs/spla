[![CI status](https://github.com/eth-cscs/spla/workflows/CI/badge.svg)](https://github.com/eth-cscs/spla/actions?query=workflow%3ACI)
[![Documentation Status](https://readthedocs.org/projects/spla/badge/?version=latest)](https://spla.readthedocs.io/en/latest/?badge=latest)

# SPLA - Specialized Parallel Linear Algebra
SPLA provides specialized functions for linear algebra computations with a C++ and C interface, which are inspired by requirements in computational material science codes.

Currently, SPLA provides functions for distributed matrix multiplications with specific matrix distributions, which cannot be used directly with a ScaLAPACK interface.
All computations can optionally utilize GPUs through CUDA or ROCm, where matrices can be located either in host or device memory.

## GEMM
The function `gemm(...)` computes a local general matrix product, that works similar to cuBLASXt. If GPU support is enabled, the function may take any combination of host and device pointer. In addition, it may use custom multi-threading for host computations, if the provided BLAS library does not support multi-threading.

### Stripe-Stripe-Block
The `pgemm_ssb(...)` function computes

![ethz](docs/images/ssb_formula.svg)

where matrices A and B are stored in a "stripe" distribution with variable block length. Matrix C can be in any supported block distribution, including the block-cyclic ScaLAPACK layout. Matrix A may be read as transposed or conjugate transposed.


                     ------ T     ------
                     |    |       |    |
                     |    |       |    |
                     ------       ------
     -------         |    |       |    |        -------
     |  |  |         |    |       |    |        |  |  |
     -------   <--   ------   *   ------    +   -------
     |  |  |         |    |       |    |        |  |  |
     -------         |    |       |    |        -------
        C            ------       ------           C
                     |    |       |    |
                     |    |       |    |
                     ------       ------
                       A            B



### Stripe-Block-Stripe
The `pgemm_sbs(...)` function computes

![ethz](docs/images/sbs_formula.svg)

where matrices A and C are stored in a "stripe" distribution with variable block length. Matrix B can be in any supported block distribution, including the block-cyclic ScaLAPACK layout.

     ------         ------                     ------
     |    |         |    |                     |    |
     |    |         |    |                     |    |
     ------         ------                     ------
     |    |         |    |       -------       |    |
     |    |         |    |       |  |  |       |    |
     ------   <--   ------   *   -------   +   ------
     |    |         |    |       |  |  |       |    |
     |    |         |    |       -------       |    |
     ------         ------          B          ------
     |    |         |    |                     |    |
     |    |         |    |                     |    |
     ------         ------                     ------
       C               A                         C

## Documentation
Documentation can be found [here](https://spla.readthedocs.io/en/latest/).

## Installation
The build system follows the standard CMake workflow. Example:
```console
mkdir build
cd build
cmake .. -DSPLA_OMP=ON -DSPLA_GPU_BACKEND=CUDA -DCMAKE_INSTALL_PREFIX=${path_to_install_to}
make -j8 install
```

### CMake options
| Option                | Default | Description                                      |
|-----------------------|---------|--------------------------------------------------|
| SPLA_OMP              | ON      | Enable multi-threading with OpenMP               |
| SPLA_GPU_BACKEND      | OFF     | Select GPU backend. Can be OFF, CUDA or ROCM     |
| SPLA_BUILD_TESTS      | OFF     | Build test executables for developement purposes |
| SPLA_INSTALL          | ON      | Add library to install target                    |

## Acknowledgements
This work was supported by:


|![ethz](docs/images/logo_ethz.png) | [**Swiss Federal Institute of Technology in Zurich**](https://www.ethz.ch/) |
|:----:|:----:|
|![cscs](docs/images/logo_cscs.png) | [**Swiss National Supercomputing Centre**](https://www.cscs.ch/)            |
|![max](docs/images/logo_max.png)  | [**MAterials design at the eXascale**](http://www.max-centre.eu) <br> (Horizon2020, grant agreement MaX CoE, No. 824143) |
