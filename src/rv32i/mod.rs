use std::fmt;

use crate::cim::{Program as CimProgram, TileDispatch};

mod optimizer;

const MMIO_VMM_BASE: u32 = 0x1000_0000;
const ACCUM_BUF_BASE: u32 = 0x2000_0000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imm {
    Dec(i32),
    Hex(u32),
}

impl fmt::Display for Imm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Imm::Dec(value) => write!(f, "{value}"),
            Imm::Hex(value) => write!(f, "{value:#x}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    X0,
    T0,
    T1,
    T2,
    T3,
    T4,
    T5,
    A0,
    A1,
    A7,
}

impl fmt::Display for Reg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reg::X0 => write!(f, "x0"),
            Reg::T0 => write!(f, "t0"),
            Reg::T1 => write!(f, "t1"),
            Reg::T2 => write!(f, "t2"),
            Reg::T3 => write!(f, "t3"),
            Reg::T4 => write!(f, "t4"),
            Reg::T5 => write!(f, "t5"),
            Reg::A0 => write!(f, "a0"),
            Reg::A1 => write!(f, "a1"),
            Reg::A7 => write!(f, "a7"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    Li(Reg, Imm),
    Lw(Reg, Imm, Reg),
    Sw(Reg, Imm, Reg),
    Add(Reg, Reg, Reg),
    Addi(Reg, Reg, i32),
    Andi(Reg, Reg, Imm),
    Srli(Reg, Reg, u32),
    Slli(Reg, Reg, u32),
    Blt(Reg, Reg, String),
    Ecall,
    Label(String),
    Directive(String),
    Blank,
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instr::Li(rd, imm) => write!(f, "    li   {rd}, {imm}"),
            Instr::Lw(rd, off, base) => write!(f, "    lw   {rd}, {off}({base})"),
            Instr::Sw(rs2, off, base) => write!(f, "    sw   {rs2}, {off}({base})"),
            Instr::Add(rd, rs1, rs2) => write!(f, "    add  {rd}, {rs1}, {rs2}"),
            Instr::Addi(rd, rs1, imm) => write!(f, "    addi {rd}, {rs1}, {imm}"),
            Instr::Andi(rd, rs1, imm) => write!(f, "    andi {rd}, {rs1}, {imm}"),
            Instr::Srli(rd, rs1, shamt) => write!(f, "    srli {rd}, {rs1}, {shamt}"),
            Instr::Slli(rd, rs1, shamt) => write!(f, "    slli {rd}, {rs1}, {shamt}"),
            Instr::Blt(rs1, rs2, label) => write!(f, "    blt  {rs1}, {rs2}, {label}"),
            Instr::Ecall => write!(f, "    ecall"),
            Instr::Label(name) => write!(f, "{name}:"),
            Instr::Directive(directive) => write!(f, "{directive}"),
            Instr::Blank => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssemblyProgram {
    pub instrs: Vec<Instr>,
}

impl AssemblyProgram {
    pub fn render(&self) -> String {
        self.instrs
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }
}

pub fn lower_to_ast(program: &CimProgram) -> Result<AssemblyProgram, String> {
    program.verify()?;
    let mut instrs = Vec::new();
    instrs.extend(prologue(program.entry.dispatches.len()));
    instrs.extend(loop_body());
    instrs.extend(epilogue());
    optimizer::peephole(&mut instrs);
    Ok(AssemblyProgram { instrs })
}

pub fn emit_assembly(program: &CimProgram) -> Result<String, String> {
    let ast = lower_to_ast(program)?;
    Ok(format!("{}\n{}", file_header(program), ast.render()))
}

fn li(rd: Reg, imm: i32) -> Instr {
    Instr::Li(rd, Imm::Dec(imm))
}

fn li_hex(rd: Reg, imm: u32) -> Instr {
    Instr::Li(rd, Imm::Hex(imm))
}

fn lw(rd: Reg, offset: Imm, base: Reg) -> Instr {
    Instr::Lw(rd, offset, base)
}

fn sw(rs2: Reg, offset: Imm, base: Reg) -> Instr {
    Instr::Sw(rs2, offset, base)
}

fn add(rd: Reg, rs1: Reg, rs2: Reg) -> Instr {
    Instr::Add(rd, rs1, rs2)
}

fn addi(rd: Reg, rs1: Reg, imm: i32) -> Instr {
    Instr::Addi(rd, rs1, imm)
}

fn andi(rd: Reg, rs1: Reg, imm: Imm) -> Instr {
    Instr::Andi(rd, rs1, imm)
}

fn srli(rd: Reg, rs1: Reg, shamt: u32) -> Instr {
    Instr::Srli(rd, rs1, shamt)
}

fn slli(rd: Reg, rs1: Reg, shamt: u32) -> Instr {
    Instr::Slli(rd, rs1, shamt)
}

fn blt(rs1: Reg, rs2: Reg, label: &str) -> Instr {
    Instr::Blt(rs1, rs2, label.to_string())
}

use Reg::*;

fn prologue(num_tiles: usize) -> Vec<Instr> {
    vec![
        Instr::Directive(".section .text".to_string()),
        Instr::Directive(".global _start".to_string()),
        Instr::Blank,
        Instr::Label("_start".to_string()),
        li(T0, 0),
        li(T1, num_tiles as i32),
        li_hex(T2, MMIO_VMM_BASE),
        li_hex(T3, ACCUM_BUF_BASE),
    ]
}

fn loop_body() -> Vec<Instr> {
    vec![
        Instr::Blank,
        Instr::Label(".loop".to_string()),
        sw(T0, Imm::Hex(0x0), T2),
        lw(A0, Imm::Hex(0x4), T2),
        andi(T4, T0, Imm::Hex(0xF)),
        srli(T4, T4, 2),
        slli(T4, T4, 8),
        add(T5, T3, T4),
        lw(A1, Imm::Hex(0x0), T5),
        add(A1, A1, A0),
        sw(A1, Imm::Hex(0x0), T5),
        addi(T0, T0, 1),
        blt(T0, T1, ".loop"),
    ]
}

fn epilogue() -> Vec<Instr> {
    vec![Instr::Blank, li(A0, 0), li(A7, 93), Instr::Ecall]
}

fn file_header(program: &CimProgram) -> String {
    let mut dispatches: Vec<&TileDispatch> = program.entry.dispatches.iter().collect();
    dispatches.sort_by_key(|op| op.order);

    let mut lines = vec![
        "# cim-compile generated - RV32I orchestration loop".to_string(),
        format!(
            "# {} tiles  (derived from verified cim dialect)",
            dispatches.len()
        ),
        "# target: RV32I simulator".to_string(),
        "#".to_string(),
        "# MMIO map:".to_string(),
        format!(
            "#   {:#010x}  VMM control reg  (sw tile_index triggers crossbar)",
            MMIO_VMM_BASE
        ),
        format!(
            "#   {:#010x}  VMM result reg   (lw reads ADC output)",
            MMIO_VMM_BASE + 4
        ),
        format!("#   {:#010x}  accumulator buffer base", ACCUM_BUF_BASE),
        "#".to_string(),
        "# Tile schedule:".to_string(),
    ];

    for op in dispatches {
        lines.push(format!(
            "#   [{:>2}]  {}  tile[{},{}]  weights@{}",
            op.order, op.projection, op.tile.row, op.tile.col, op.weight_offset
        ));
    }
    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cim::{
        MatrixShape, Program, ProjectionKind, TileCoord, TileDispatch, TileSize,
        expected_weight_offset,
    };

    fn tiny_program() -> Program {
        let tile_size = TileSize::new(1, 1);
        Program::new(
            "test",
            "main",
            vec![TileDispatch {
                projection: ProjectionKind::Main,
                tile: TileCoord::new(0, 0),
                matrix_shape: MatrixShape::new(1, 1),
                tile_size,
                weight_offset: expected_weight_offset(0, tile_size),
                order: 0,
            }],
        )
    }

    #[test]
    fn lower_to_ast_uses_typed_instructions() {
        let ast = lower_to_ast(&tiny_program()).unwrap();

        assert!(matches!(ast.instrs[0], Instr::Directive(_)));
        assert!(
            ast.instrs
                .iter()
                .any(|instr| matches!(instr, Instr::Sw(T0, Imm::Hex(0), T2)))
        );
    }

    #[test]
    fn assembly_renderer_emits_loop() {
        let asm = emit_assembly(&tiny_program()).unwrap();

        assert!(asm.contains(".global _start"));
        assert!(asm.contains("blt  t0, t1, .loop"));
        assert!(asm.contains("andi t4, t0, 0xc"));
    }
}
