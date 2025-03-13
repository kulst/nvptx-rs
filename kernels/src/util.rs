#[inline]
pub fn get_init_worker_count(block_dim: usize) -> usize {
    if block_dim == 1 {
        return 0;
    }
    let mut i = 1;
    while i < 512 {
        if i * 2 >= block_dim {
            return i;
        }
        i *= 2;
    }
    return 512;
}
