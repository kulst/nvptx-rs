#[derive(Clone, Copy)]
pub struct Linear2D<T> {
    base: *const T,
    width: isize,
    _height: isize,
}

impl<T: Copy> Linear2D<T> {
    pub fn new(base: *const T, width: isize, height: isize) -> Self {
        Self {
            base,
            width,
            _height: height,
        }
    }

    #[inline]
    pub unsafe fn get(&self, col: isize, row: isize) -> T {
        let index = col + row * self.width;
        self.base.offset(index).read()
    }

    #[inline]
    pub unsafe fn set(&mut self, val: T, col: isize, row: isize) {
        let index = col + row * self.width;
        self.base.cast_mut().offset(index).write(val)
    }
}
pub struct Linear3D<T> {
    base: *const T,
    width: isize,
    height: isize,
    _depth: isize,
}

impl<T: Copy> Linear3D<T> {
    pub fn new(base: *const T, width: isize, height: isize, depth: isize) -> Self {
        Self {
            base,
            width,
            height,
            _depth: depth,
        }
    }

    #[inline]
    pub unsafe fn get(&self, col: isize, row: isize, plane: isize) -> T {
        let index = col + row * self.width + plane * self.width * self.height;
        self.base.offset(index).read()
    }

    #[inline]
    pub unsafe fn set(&mut self, val: T, col: isize, row: isize, plane: isize) {
        let index = col + row * self.width + plane * self.width * self.height;
        self.base.cast_mut().offset(index).write(val)
    }
}

pub struct Linear4D<T> {
    base: *const T,
    width: isize,
    height: isize,
    depth: isize,
    _num: isize,
}

impl<T: Copy> Linear4D<T> {
    pub fn new(base: *const T, width: isize, height: isize, depth: isize, num: isize) -> Self {
        Self {
            base,
            width,
            height,
            depth,
            _num: num,
        }
    }

    #[inline]
    pub unsafe fn get(&self, col: isize, row: isize, plane: isize, id: isize) -> T {
        let index = col
            + row * self.width
            + plane * self.width * self.height
            + id * self.width * self.height * self.depth;
        self.base.offset(index).read()
    }

    #[inline]
    pub unsafe fn set(&mut self, val: T, col: isize, row: isize, plane: isize, id: isize) {
        let index = col
            + row * self.width
            + plane * self.width * self.height
            + id * self.width * self.height * self.depth;
        self.base.cast_mut().offset(index).write(val);
    }
}
