pub struct CrossbarSpec {
    pub tile_rows: u32,
    pub tile_cols: u32,
}

impl CrossbarSpec {
    pub fn new(size: u32) -> Self {
        CrossbarSpec {
            tile_rows: size,
            tile_cols: size,
        }
    }

}
