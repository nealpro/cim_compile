pub struct CrossbarSpec {
    pub tile_rows: u32,
    pub tile_cols: u32,
}

impl CrossbarSpec {
    pub fn default_128x128() -> Self {
        CrossbarSpec {
            tile_rows: 128,
            tile_cols: 128,
        }
    }
}
