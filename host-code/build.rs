fn main() {
    println!("cargo::rerun-if-changed=../kernels/target/nvptx64-nvidia-cuda/release/kernels.ptx");
}
