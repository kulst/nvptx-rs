use core::any::TypeId;
use core::arch::{asm, global_asm};
use core::hint::unreachable_unchecked;

use num::FromPrimitive;
use num::{traits::float::FloatCore, ToPrimitive};

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

global_asm!(".extern .shared .align 8 .b8 rust_cuda_dynamic_shared_base[];");

pub struct DynSmem {
    base: *mut u8,
}

impl DynSmem {
    // SAFETY must only be called once per thread per thread block
    pub unsafe fn new() -> Self {
        let base: *mut u8;
        unsafe {
            asm!(
            "cvta.shared.u64 {base}, rust_cuda_dynamic_shared_base;",
            base = out(reg64) base,
            );
        }
        Self { base }
    }

    pub unsafe fn get_chunk<T>(smem: *mut Self, len: usize) -> *mut T {
        // calculate the aligned base pointer for the type to return
        let base = (*smem).base;
        let aligned_base = base.byte_add(base.align_offset(core::mem::align_of::<T>()));
        // casting the aligned base pointer to the type to return
        let data: *mut T = aligned_base.cast();
        // calculate the new base pointer of dynamic shared memory
        (*smem).base = data.add(len).cast::<u8>();
        // writing the new base pointer of the dynamic shared memory to a global register
        data
    }
}

#[inline]
pub unsafe fn _atomic_add<T>(address: *mut T, val: T) -> T
where
    T: FloatCore + 'static + ToPrimitive + FromPrimitive,
{
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        let old: f32;
        let val = val.to_f32().unwrap_unchecked();
        unsafe {
            asm!(
                "atom.add.f32 {old}, [{address}], {val};",
                old = out(reg32) old,
                address = in(reg64) address,
                val = in(reg32) val,
            );
        }
        T::from_f32(old).unwrap_unchecked()
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        let old: f64;
        let val = val.to_f64().unwrap_unchecked();
        unsafe {
            asm!(
                "atom.add.f64 {old}, [{address}], {val};",
                old = out(reg64) old,
                address = in(reg64) address,
                val = in(reg64) val,
            );
        }
        T::from_f64(old).unwrap_unchecked()
    } else {
        unreachable_unchecked()
    }
}
