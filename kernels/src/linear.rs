#[derive(Clone, Copy)]
pub struct Linear2D<T> {
    base: *const T,
    width: usize,
    _height: usize,
}

impl<T: Copy> Linear2D<T> {
    pub fn new(base: *const T, width: usize, height: usize) -> Self {
        Self {
            base,
            width,
            _height: height,
        }
    }

    #[inline]
    pub unsafe fn get(&self, col: usize, row: usize) -> T {
        let index = col + row * self.width;
        self.base.add(index).read()
    }

    #[inline]
    pub unsafe fn set(&mut self, val: T, col: usize, row: usize) {
        let index = col + row * self.width;
        self.base.cast_mut().add(index).write(val)
    }
}
pub struct Linear3D<T> {
    base: *const T,
    width: usize,
    height: usize,
    _depth: usize,
}

impl<T: Copy> Linear3D<T> {
    pub fn new(base: *const T, width: usize, height: usize, depth: usize) -> Self {
        Self {
            base,
            width,
            height,
            _depth: depth,
        }
    }

    #[inline]
    pub unsafe fn get(&self, col: usize, row: usize, plane: usize) -> T {
        let index = col + row * self.width + plane * self.width * self.height;
        self.base.add(index).read()
    }

    #[inline]
    pub unsafe fn set(&mut self, val: T, col: usize, row: usize, plane: usize) {
        let index = col + row * self.width + plane * self.width * self.height;
        self.base.cast_mut().add(index).write(val)
    }
}
