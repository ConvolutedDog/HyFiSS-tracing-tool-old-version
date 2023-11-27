## Tracing Tool

The ***tracing_tool*** is used to extract the memory and SASS traces. This repo use and extend NVBit (NVidia Binary Instrumentation Tool) which is a research prototype of a dynamic binary instrumentation library for NVIDIA GPUs. Licence and agreement of NVBIT is found in the origianal [NVBIT repo](https://github.com/NVlabs/NVBit) ("This software contains source code provided by NVIDIA Corporation”)

NVBIT does not require application source code, any pre-compiled GPU application should work regardless of which compiler (or version) has been used (i.e. nvcc, pgicc, etc).

## Usage

* We will further provide tutorial for how to build the tracing tool and extract the traces using docker.

*  Setup the **MAX_KERNELS** variable in **tracer.cu** to define the limit on the number of kernels you want to instrument in the application 

* For stanalone building and running of the tracing_tool (no docker), please see below: 

  #### Building the tool
  
  * Setup **ARCH** variable in the Makefile
  * run make clean; make

  #### Extracting the traces
  
  ```
  LD_PRELOAD=/path_to_tracing_tool/tracer.so ./app
  ```
  
  The above command outputs two folders **memory_traces** and **sass_traces** each has the applications kernel traces. It also output **configs** file whoch has information about the kernel executing inside the application. 
