use crate::intrinsics::_tex_1d_fetch_f32;

#[repr(C)]
pub struct TexObjectF32_3D {
    inner: u64,
    width: i32,
    height: i32,
    depth: i32,
}

impl TexObjectF32_3D {
    pub fn new(inner: u64, width: i32, height: i32, depth: i32) -> Self {
        Self {
            inner,
            width,
            height,
            depth,
        }
    }

    #[inline]
    pub unsafe fn get(&mut self, col: i32, row: i32, plane: i32) -> f32 {
        let index = col + row * self.width + plane * self.width * self.height;
        _tex_1d_fetch_f32(self.inner, index)
    }
}
