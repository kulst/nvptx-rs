use core::arch::{asm, global_asm};

#[inline]
pub unsafe fn _tex_1d_fetch_f32(tex_object: u64, x: i32) -> f32 {
    let mut res: f32;
    asm! {
        "tex.1d.v4.f32.s32 {{{f1}, {f2}, {f3}, {f4}}}, [{tex_object}, {{{x}}}];",
        tex_object = in(reg64) tex_object,
        x = in(reg32) x,
        f1 = out(reg32) res,
        f2 = out(reg32) _,
        f3 = out(reg32) _,
        f4 = out(reg32) _,
    }
    res
}

#[inline]
pub unsafe fn _static_shared_mem<T>() -> *mut T {
    let shared: *mut T;

    unsafe {
        core::arch::asm!(
            ".shared .align {align} .b8 {reg}_rust_cuda_static_shared[{size}];",
            "cvta.shared.u64 {reg}, {reg}_rust_cuda_static_shared;",
            reg = out(reg64) shared,
            align = const(core::mem::align_of::<T>()),
            size = const(core::mem::size_of::<T>()),
        );
    }
    shared
}

/// # Safety
///
/// The thread-block shared dynamic memory must be initialised once and
/// only once per kernel.
#[inline]
pub unsafe fn _init_dyn_shared_mem() {
    unsafe {
        asm!(".reg.u64 %rust_cuda_dynamic_shared;");
        asm!("cvta.shared.u64 %rust_cuda_dynamic_shared, rust_cuda_dynamic_shared_base;",);
    }
}

global_asm!(".extern .shared .align 8 .b8 rust_cuda_dynamic_shared_base[];");

#[inline]
pub unsafe fn _dynamic_shared_mem<T>(len: usize) -> *mut T {
    let base: *mut u8;
    // we read the current dyn shared memory base pointer
    unsafe {
        core::arch::asm!(
            "mov.u64    {base}, %rust_cuda_dynamic_shared;",
            base = out(reg64) base,
        );
    }
    // calculate the aligned base pointer for the type to return
    let aligned_base = base.byte_add(base.align_offset(core::mem::align_of::<T>()));
    // casting the aligned base pointer to the type to return
    let data: *mut T = aligned_base.cast();
    // calculate the new base pointer of dynamic shared memory
    let new_base = data.add(len).cast::<u8>();
    // writing the new base pointer of the dynamic shared memory to a global register
    unsafe {
        core::arch::asm!(
            "mov.u64    %rust_cuda_dynamic_shared, {new_base};",
            new_base = in(reg64) new_base,
        );
    }
    data
}

#[inline]
pub unsafe fn _atomic_add_f32(address: *mut f32, val: f32) -> f32 {
    let old: f32;
    unsafe {
        asm!(
            "atom.add.f32 {old}, [{address}], {val};",
            old = out(reg32) old,
            address = in(reg64) address,
            val = in(reg32) val,
        );
    }
    old
}
