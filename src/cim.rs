use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ProjectionKind {
    Main,
    WQ,
    WK,
    WV,
    WO,
    Named(String),
}

impl ProjectionKind {
    pub fn as_str(&self) -> Cow<'_, str> {
        match self {
            ProjectionKind::Main => Cow::Borrowed("main"),
            ProjectionKind::WQ => Cow::Borrowed("wq"),
            ProjectionKind::WK => Cow::Borrowed("wk"),
            ProjectionKind::WV => Cow::Borrowed("wv"),
            ProjectionKind::WO => Cow::Borrowed("wo"),
            ProjectionKind::Named(name) => Cow::Borrowed(name),
        }
    }
}

impl fmt::Display for ProjectionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

impl FromStr for ProjectionKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let value = value.trim_matches('"');
        match value {
            "main" => Ok(ProjectionKind::Main),
            "wq" | "WQ" => Ok(ProjectionKind::WQ),
            "wk" | "WK" => Ok(ProjectionKind::WK),
            "wv" | "WV" => Ok(ProjectionKind::WV),
            "wo" | "WO" => Ok(ProjectionKind::WO),
            other if is_projection_name(other) => Ok(ProjectionKind::Named(other.to_string())),
            other => Err(format!("invalid projection name `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatrixShape {
    pub rows: u32,
    pub cols: u32,
}

impl MatrixShape {
    pub fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileSize {
    pub rows: u32,
    pub cols: u32,
}

impl TileSize {
    pub fn new(rows: u32, cols: u32) -> Self {
        Self { rows, cols }
    }

    pub fn payload_bytes(self) -> u64 {
        self.rows as u64 * self.cols as u64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TileCoord {
    pub row: u32,
    pub col: u32,
}

impl TileCoord {
    pub fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TileDispatch {
    pub projection: ProjectionKind,
    pub tile: TileCoord,
    pub matrix_shape: MatrixShape,
    pub tile_size: TileSize,
    pub weight_offset: u64,
    pub quant_scale: f32,
    pub order: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub name: String,
    pub dispatches: Vec<TileDispatch>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub module_name: String,
    pub entry: Function,
}

impl Program {
    pub fn new(
        module_name: impl Into<String>,
        entry_name: impl Into<String>,
        dispatches: Vec<TileDispatch>,
    ) -> Self {
        Self {
            module_name: module_name.into(),
            entry: Function {
                name: entry_name.into(),
                dispatches,
            },
        }
    }

    pub fn verify(&self) -> Result<(), String> {
        verify_program(self)
    }

    pub fn to_text(&self) -> String {
        print_program(self)
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_text())
    }
}

impl FromStr for Program {
    type Err = String;

    fn from_str(text: &str) -> Result<Self, Self::Err> {
        parse_program(text)
    }
}

pub fn expected_weight_offset(order: u32, tile_size: TileSize) -> u64 {
    order as u64 * tile_size.payload_bytes()
}

fn is_projection_name(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == '_')
        && chars.all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

pub fn parse_program(text: &str) -> Result<Program, String> {
    let mut module_name = None;
    let mut entry_name = None;
    let mut dispatches = Vec::new();

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix("cim.module @") {
            module_name = Some(parse_symbol_header(rest, "module")?);
        } else if let Some(rest) = line.strip_prefix("cim.func @") {
            entry_name = Some(parse_symbol_header(rest, "function")?);
        } else if line.starts_with("cim.tile.dispatch") {
            dispatches.push(parse_dispatch_line(line)?);
        } else if line == "{" || line == "}" {
            continue;
        } else {
            return Err(format!("unsupported cim dialect line `{line}`"));
        }
    }

    let program = Program::new(
        module_name.ok_or_else(|| "missing cim.module header".to_string())?,
        entry_name.ok_or_else(|| "missing cim.func header".to_string())?,
        dispatches,
    );
    program.verify()?;
    Ok(program)
}

fn parse_symbol_header(rest: &str, kind: &str) -> Result<String, String> {
    let name = rest
        .trim()
        .strip_suffix('{')
        .ok_or_else(|| format!("malformed {kind} header"))?
        .trim();
    if name.is_empty() {
        Err(format!("{kind} name cannot be empty"))
    } else {
        Ok(name.to_string())
    }
}

fn parse_dispatch_line(line: &str) -> Result<TileDispatch, String> {
    let attrs = line
        .strip_prefix("cim.tile.dispatch")
        .and_then(|s| s.trim().strip_prefix('{'))
        .and_then(|s| s.trim().strip_suffix('}'))
        .ok_or_else(|| "malformed cim.tile.dispatch attribute dictionary".to_string())?;
    let map = parse_attr_dict(attrs)?;

    Ok(TileDispatch {
        projection: parse_required::<ProjectionKind>(&map, "projection")?,
        tile: parse_pair(&map, "tile").map(|(row, col)| TileCoord::new(row, col))?,
        matrix_shape: parse_pair(&map, "matrix_shape")
            .map(|(rows, cols)| MatrixShape::new(rows, cols))?,
        tile_size: parse_pair(&map, "tile_size").map(|(rows, cols)| TileSize::new(rows, cols))?,
        weight_offset: parse_required::<u64>(&map, "weight_offset")?,
        quant_scale: parse_required::<f32>(&map, "quant_scale")?,
        order: parse_required::<u32>(&map, "order")?,
    })
}

fn parse_attr_dict(attrs: &str) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    for part in split_top_level(attrs) {
        let (key, value) = part
            .split_once('=')
            .ok_or_else(|| format!("malformed attribute `{part}`"))?;
        let key = key.trim().to_string();
        if map.insert(key.clone(), value.trim().to_string()).is_some() {
            return Err(format!("duplicate attribute `{key}`"));
        }
    }
    Ok(map)
}

fn split_top_level(attrs: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut start = 0usize;
    for (idx, ch) in attrs.char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => depth -= 1,
            ',' if depth == 0 => {
                parts.push(attrs[start..idx].trim().to_string());
                start = idx + 1;
            }
            _ => {}
        }
    }
    let tail = attrs[start..].trim();
    if !tail.is_empty() {
        parts.push(tail.to_string());
    }
    parts
}

fn parse_required<T>(map: &BTreeMap<String, String>, key: &str) -> Result<T, String>
where
    T: FromStr,
    T::Err: fmt::Display,
{
    map.get(key)
        .ok_or_else(|| format!("missing `{key}` attribute"))?
        .trim_matches('"')
        .parse::<T>()
        .map_err(|err| format!("invalid `{key}` attribute: {err}"))
}

fn parse_pair(map: &BTreeMap<String, String>, key: &str) -> Result<(u32, u32), String> {
    let raw = map
        .get(key)
        .ok_or_else(|| format!("missing `{key}` attribute"))?;
    let body = raw
        .trim()
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("`{key}` must be a two-element list"))?;
    let mut values = body.split(',').map(str::trim);
    let first = values
        .next()
        .ok_or_else(|| format!("`{key}` must have two elements"))?
        .parse::<u32>()
        .map_err(|err| format!("invalid `{key}` first element: {err}"))?;
    let second = values
        .next()
        .ok_or_else(|| format!("`{key}` must have two elements"))?
        .parse::<u32>()
        .map_err(|err| format!("invalid `{key}` second element: {err}"))?;
    if values.next().is_some() {
        return Err(format!("`{key}` must have exactly two elements"));
    }
    Ok((first, second))
}

fn print_program(program: &Program) -> String {
    let mut dispatches = program.entry.dispatches.clone();
    dispatches.sort_by_key(|op| op.order);

    let mut out = String::new();
    out.push_str(&format!("cim.module @{} {{\n", program.module_name));
    out.push_str(&format!("  cim.func @{} {{\n", program.entry.name));
    for op in dispatches {
        out.push_str(&format!(
            "    cim.tile.dispatch {{ projection = \"{}\", tile = [{}, {}], matrix_shape = [{}, {}], tile_size = [{}, {}], weight_offset = {}, quant_scale = {:?}, order = {} }}\n",
            op.projection,
            op.tile.row,
            op.tile.col,
            op.matrix_shape.rows,
            op.matrix_shape.cols,
            op.tile_size.rows,
            op.tile_size.cols,
            op.weight_offset,
            op.quant_scale,
            op.order
        ));
    }
    out.push_str("  }\n");
    out.push_str("}\n");
    out
}

fn verify_program(program: &Program) -> Result<(), String> {
    if program.module_name.trim().is_empty() {
        return Err("module name cannot be empty".to_string());
    }
    if program.entry.name.trim().is_empty() {
        return Err("entry function name cannot be empty".to_string());
    }
    if program.entry.dispatches.is_empty() {
        return Err("entry function must contain at least one cim.tile.dispatch".to_string());
    }

    let global_tile_size = program.entry.dispatches[0].tile_size;

    let dispatch_count = program.entry.dispatches.len();
    let mut seen_orders = vec![false; dispatch_count];
    let mut seen_tiles = BTreeSet::new();
    let mut by_projection: BTreeMap<ProjectionKind, BTreeSet<TileCoord>> = BTreeMap::new();
    let mut projection_shapes: BTreeMap<ProjectionKind, MatrixShape> = BTreeMap::new();

    for op in &program.entry.dispatches {
        if op.tile_size != global_tile_size {
            return Err("all dispatches in a program must use the same tile_size".to_string());
        }
        validate_shape(op.matrix_shape, op.tile_size)?;
        if let Some(previous_shape) =
            projection_shapes.insert(op.projection.clone(), op.matrix_shape)
            && previous_shape != op.matrix_shape
        {
            return Err(format!(
                "projection {} uses inconsistent matrix_shape values",
                op.projection
            ));
        }
        if !op.quant_scale.is_finite() || op.quant_scale <= 0.0 {
            return Err(format!(
                "dispatch order {} has invalid quant_scale {}",
                op.order, op.quant_scale
            ));
        }
        if op.order as usize >= dispatch_count {
            return Err(format!(
                "dispatch order {} is out of range for {} dispatches",
                op.order, dispatch_count
            ));
        }
        if seen_orders[op.order as usize] {
            return Err(format!("duplicate dispatch order {}", op.order));
        }
        seen_orders[op.order as usize] = true;

        let row_tiles = op.matrix_shape.rows.div_ceil(op.tile_size.rows);
        let col_tiles = op.matrix_shape.cols.div_ceil(op.tile_size.cols);
        if op.tile.row >= row_tiles || op.tile.col >= col_tiles {
            return Err(format!(
                "tile [{}, {}] is out of bounds for {}x{} tiles",
                op.tile.row, op.tile.col, row_tiles, col_tiles
            ));
        }

        let expected_offset = expected_weight_offset(op.order, op.tile_size);
        if op.weight_offset != expected_offset {
            return Err(format!(
                "dispatch order {} has weight_offset {}, expected {}",
                op.order, op.weight_offset, expected_offset
            ));
        }

        if !seen_tiles.insert((op.projection.clone(), op.tile)) {
            return Err(format!(
                "duplicate tile coverage for projection {} tile [{}, {}]",
                op.projection, op.tile.row, op.tile.col
            ));
        }
        by_projection
            .entry(op.projection.clone())
            .or_default()
            .insert(op.tile);
    }

    if let Some(missing) = seen_orders.iter().position(|seen| !*seen) {
        return Err(format!("missing dispatch order {missing}"));
    }

    for (projection, tiles) in &by_projection {
        let shape = projection_shapes
            .get(projection)
            .expect("projection shape exists for covered tiles");
        let expected_tiles =
            shape.rows.div_ceil(global_tile_size.rows) * shape.cols.div_ceil(global_tile_size.cols);
        if tiles.len() != expected_tiles as usize {
            return Err(format!(
                "projection {projection} covers {} tiles, expected {}",
                tiles.len(),
                expected_tiles
            ));
        }
    }

    let mut canonical = program.entry.dispatches.clone();
    canonical.sort_by_key(|op| (op.projection.clone(), op.tile.row, op.tile.col));
    for (expected_order, op) in canonical.iter().enumerate() {
        if op.order as usize != expected_order {
            return Err(format!(
                "inconsistent dispatch schedule for projection {} tile [{}, {}]: order {}, expected {}",
                op.projection, op.tile.row, op.tile.col, op.order, expected_order
            ));
        }
    }

    Ok(())
}

fn validate_shape(shape: MatrixShape, tile_size: TileSize) -> Result<(), String> {
    if shape.rows == 0 || shape.cols == 0 {
        return Err("matrix_shape dimensions must be greater than zero".to_string());
    }
    if tile_size.rows == 0 || tile_size.cols == 0 {
        return Err("tile_size dimensions must be greater than zero".to_string());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch(order: u32, row: u32, col: u32) -> TileDispatch {
        TileDispatch {
            projection: ProjectionKind::Main,
            tile: TileCoord::new(row, col),
            matrix_shape: MatrixShape::new(2, 2),
            tile_size: TileSize::new(1, 1),
            weight_offset: expected_weight_offset(order, TileSize::new(1, 1)),
            quant_scale: 1.0,
            order,
        }
    }

    fn tiny_program() -> Program {
        Program::new(
            "test",
            "main",
            vec![
                dispatch(0, 0, 0),
                dispatch(1, 0, 1),
                dispatch(2, 1, 0),
                dispatch(3, 1, 1),
            ],
        )
    }

    #[test]
    fn verifier_accepts_complete_canonical_schedule() {
        tiny_program().verify().unwrap();
    }

    #[test]
    fn verifier_rejects_missing_order() {
        let mut program = tiny_program();
        program.entry.dispatches[3].order = 2;
        let err = program.verify().unwrap_err();
        assert!(err.contains("duplicate dispatch order"));
    }

    #[test]
    fn verifier_rejects_out_of_bounds_tile() {
        let mut program = tiny_program();
        program.entry.dispatches[0].tile.row = 2;
        let err = program.verify().unwrap_err();
        assert!(err.contains("out of bounds"));
    }

    #[test]
    fn verifier_rejects_bad_weight_offset() {
        let mut program = tiny_program();
        program.entry.dispatches[0].weight_offset = 99;
        let err = program.verify().unwrap_err();
        assert!(err.contains("weight_offset"));
    }

    #[test]
    fn verifier_rejects_inconsistent_schedule() {
        let mut program = tiny_program();
        program.entry.dispatches[0].tile = TileCoord::new(0, 1);
        program.entry.dispatches[1].tile = TileCoord::new(0, 0);
        let err = program.verify().unwrap_err();
        assert!(err.contains("inconsistent dispatch schedule"));
    }

    #[test]
    fn printer_parser_round_trip() {
        let program = tiny_program();
        let text = program.to_text();
        let parsed = parse_program(&text).unwrap();
        assert_eq!(parsed, program);
    }
}
