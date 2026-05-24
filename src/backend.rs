use crate::middle::{LowLevelOp, Projection};
use std::fmt;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

mod optimizer;

const MMIO_VMM_BASE: u32 = 0x1000_0000;
const ACCUM_BUF_BASE: u32 = 0x2000_0000;

// ── Immediate operand ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imm {
    Dec(i32),
    Hex(u32),
}

impl fmt::Display for Imm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Imm::Dec(n) => write!(f, "{n}"),
            Imm::Hex(n) => write!(f, "{n:#x}"),
        }
    }
}

// ── Register file ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reg {
    _X0,
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
            Reg::_X0 => write!(f, "x0"),
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

// ── Instruction AST ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Instr {
    Li(Reg, Imm),
    Lw(Reg, Imm, Reg),
    Sw(Reg, Imm, Reg),
    Add(Reg, Reg, Reg),
    Addi(Reg, Reg, i32),
    Andi(Reg, Reg, Imm),
    Srli(Reg, Reg, u32),
    Slli(Reg, Reg, u32),
    Blt(Reg, Reg, &'static str),
    Ecall,
    Label(&'static str),
    Directive(&'static str),
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
            Instr::Directive(d) => write!(f, "{d}"),
            Instr::Blank => Ok(()),
        }
    }
}

// ── Constructor functions ────────────────────────────────────────────────────

fn li(rd: Reg, imm: i32) -> Instr {
    Instr::Li(rd, Imm::Dec(imm))
}
fn li_hex(rd: Reg, addr: u32) -> Instr {
    Instr::Li(rd, Imm::Hex(addr))
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
fn blt(rs1: Reg, rs2: Reg, label: &'static str) -> Instr {
    Instr::Blt(rs1, rs2, label)
}

use Reg::*;

// ── Section builders ─────────────────────────────────────────────────────────

fn prologue(num_tiles: usize) -> Vec<Instr> {
    vec![
        Instr::Directive(".section .text"),
        Instr::Directive(".global _start"),
        Instr::Blank,
        Instr::Label("_start"),
        li(T0, 0),
        li(T1, num_tiles as i32),
        li_hex(T2, MMIO_VMM_BASE),
        li_hex(T3, ACCUM_BUF_BASE),
    ]
}

fn loop_body() -> Vec<Instr> {
    vec![
        Instr::Blank,
        Instr::Label(".loop"),
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

// ── Render pass ──────────────────────────────────────────────────────────────

fn render(instrs: &[Instr]) -> String {
    instrs
        .iter()
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

// ── File header (comment block, not instructions) ────────────────────────────

fn file_header(ops: &[LowLevelOp]) -> String {
    let num_tiles = ops.len();
    let mut lines = vec![
        format!("# cim-compile generated — RV32I orchestration loop"),
        format!("# {} tiles  (derived from Vec<LowLevelOp>)", num_tiles),
        format!("# target: Spike / QEMU simulator"),
        format!("#"),
        format!("# MMIO map:"),
        format!(
            "#   {:#010x}  VMM control reg  (sw tile_index → triggers crossbar)",
            MMIO_VMM_BASE
        ),
        format!(
            "#   {:#010x}  VMM result reg   (lw → ADC output)",
            MMIO_VMM_BASE + 4
        ),
        format!("#   {:#010x}  accumulator buffer base", ACCUM_BUF_BASE),
        format!("#"),
        format!("# Register map:"),
        format!("#   t0  tile index (0 … {})", num_tiles),
        format!("#   t1  total tile count"),
        format!("#   t2  MMIO VMM base address"),
        format!("#   t3  accumulator buffer base"),
        format!("#   t4  scratch: row_tile byte offset"),
        format!("#   t5  scratch: accumulator slot address"),
        format!("#   a0  partial VMM result"),
        format!("#   a1  running partial sum"),
        format!("#"),
        format!("# Tile schedule:"),
    ];
    for (i, op) in ops.iter().enumerate() {
        match op {
            LowLevelOp::ProjectionTile {
                projection,
                row,
                col,
                ..
            } => {
                lines.push(format!(
                    "#   [{:>2}]  {:?}  tile[{},{}]",
                    i, projection, row, col
                ));
            }
        }
    }
    lines.join("\n")
}

// ── Public API ───────────────────────────────────────────────────────────────

pub fn emit_asm(ops: &[LowLevelOp]) -> String {
    let mut instrs: Vec<Instr> = [prologue(ops.len()), loop_body(), epilogue()].concat();
    optimizer::peephole(&mut instrs);
    format!("{}\n{}", file_header(ops), render(&instrs))
}

pub fn write_asm(ops: &[LowLevelOp], path: &Path) -> std::io::Result<()> {
    fs::write(path, emit_asm(ops))
}

// Binary weight file format:
//   [0..4]   magic        b"CiMW"
//   [4]      version      1u8
//   [5..7]   padding
//   [7]      dtype        16u8  (bfloat16)
//   [8..12]  num_tiles    u32 LE
//   [12..16] tile_rows    u32 LE
//   [16..20] tile_cols    u32 LE
//   [20..24] padding
//   per tile:
//     [0]    projection   0=WQ 1=WK 2=WV 3=WO
//     [1]    row          u8
//     [2]    col          u8
//     [3]    padding
//     [4..]  weights      tile_rows × tile_cols × 2 bytes (bfloat16 row-major)
pub fn write_weights(ops: &[LowLevelOp], path: &Path) -> std::io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);

    w.write_all(b"CiMW")?;
    w.write_all(&[1u8, 0, 0, 16u8])?;                  // version, pad, pad, dtype
    w.write_all(&(ops.len() as u32).to_le_bytes())?;
    w.write_all(&128u32.to_le_bytes())?;                // tile_rows
    w.write_all(&128u32.to_le_bytes())?;                // tile_cols
    w.write_all(&[0u8; 4])?;                            // padding to 24 bytes

    for op in ops {
        match op {
            LowLevelOp::ProjectionTile { projection, row, col, weights } => {
                w.write_all(&[proj_id(*projection), *row as u8, *col as u8, 0])?;
                w.write_all(weights)?;
            }
        }
    }

    Ok(())
}

fn proj_id(p: Projection) -> u8 {
    match p {
        Projection::WQ => 0,
        Projection::WK => 1,
        Projection::WV => 2,
        Projection::WO => 3,
    }
}
