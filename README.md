# nvptx-rs

## Parallel programming with Rust and CUDA
This repository contains applications to demonstrate the capability of 
Rust to program and run CUDA kernels. It leverages the LLVM nvptx64 backend 
as the Rust compiler *rustc* does already support it by its *nightly* toolchain.

It was built as a project for the module *Fachpraktikum Parallel Programming* 
at the FernUniversität in Hagen during winter semester 2024/25. 

## Prerequisites
To built and run the application it is necessary
- to have a CUDA capable NVIDIA GPU
- to have CUDA installed
- to have a nightly Rust toolchain with some additional tools installed

For the Rust part the easiest way is to install these by the Rust hosted tool *rustup*.

### Installing rustup
To install rustup please visit [https://rustup.rs/](https://rustup.rs/) and 
execute the shown command in a shell of your choice.

### Installing a nightly toolchain
To build for the *nvptx64* backend a nightly toolchain is necessary. It can
be installed with the following command:

```bash
rustup toolchain install nightly
```

### Installing necessary components
For building and linking some llvm tools are necessary. To install these for the
nightly toolchain you can use the following commands:
```bash
rustup +nightly target add nvptx64-nvidia-cuda
rustup +nightly component add llvm-tools llvm-bitcode-linker
```

### Installing CUDA
CUDA can be installed by following the instructions on [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). 

## Building and running 
If everything is set it is possible to build and run the applications.
First step is to navigate to the rust directory. For example if the `fapra-project` 
folder is placed directly under `~`: `cd ~/fapra-project/rust`.

The command `cargo xtask build --release` builds the device code and the host
code and places the binaries in `fapra-project/rust/target/release`.

The applications can simply be run with
```bash
target/release/stencil
target/release/reduction
target/release/matrix_multiply
```

`stencil`, `reduction` and `matrix_multiply` have parameters that can be set from command line.
To get some information about these simply call the applications with `--help`.

# License
Licensed under either of

    Apache License, Version 2.0 (LICENSE-APACHE or http://www.apache.org/licenses/LICENSE-2.0)
    MIT license (LICENSE-MIT or http://opensource.org/licenses/MIT)

at your option.




