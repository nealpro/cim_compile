use crate::cim::ProjectionKind;

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedProgram {
    pub name: String,
    pub ops: Vec<NormalizedOp>,
}

impl NormalizedProgram {
    pub fn new(name: impl Into<String>, ops: Vec<NormalizedOp>) -> Self {
        Self {
            name: name.into(),
            ops,
        }
    }

    pub fn projections(&self) -> impl Iterator<Item = &ProjectionOp> {
        self.ops.iter().filter_map(|op| match op {
            NormalizedOp::Projection(projection) => Some(projection),
            _ => None,
        })
    }

    pub fn projection_count(&self) -> usize {
        self.projections().count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormalizedOp {
    Projection(ProjectionOp),
    Reshape { name: String },
    Transpose { name: String, perm: Vec<i64> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionOp {
    pub name: String,
    pub kind: ProjectionKind,
    pub rows: u32,
    pub cols: u32,
    pub weights: Vec<f32>,
    pub bias: Option<Vec<f32>>,
}

impl ProjectionOp {
    pub fn new(
        name: impl Into<String>,
        kind: ProjectionKind,
        rows: u32,
        cols: u32,
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
    ) -> Result<Self, String> {
        let expected = rows as usize * cols as usize;
        if rows == 0 || cols == 0 {
            return Err("projection dimensions must be greater than zero".to_string());
        }
        if weights.len() != expected {
            return Err(format!(
                "projection {}x{} has {} weights, expected {}",
                rows,
                cols,
                weights.len(),
                expected
            ));
        }
        if let Some(bias) = &bias {
            if bias.len() != rows as usize {
                return Err(format!(
                    "projection bias has {} elements, expected {}",
                    bias.len(),
                    rows
                ));
            }
        }
        Ok(Self {
            name: name.into(),
            kind,
            rows,
            cols,
            weights,
            bias,
        })
    }
}
