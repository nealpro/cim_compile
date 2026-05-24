use super::{Imm, Instr};
use std::cmp::Ordering;

pub(super) fn peephole(instrs: &mut Vec<Instr>) {
    let mut result = Vec::with_capacity(instrs.len());
    let mut i = 0;
    while i < instrs.len() {
        if let Some(fold) = try_shift_fold(instrs, i) {
            result.extend(fold);
            i += 3;
        } else {
            result.push(instrs[i].clone());
            i += 1;
        }
    }
    *instrs = result;
}

// Matches: Andi(rd, rs, mask) + Srli(rd, rd, a) + Slli(rd, rd, b)
// The srli discards bits [a-1:0]; zeroing those bits in the mask makes the
// srli lossless, letting both shifts collapse into one net shift.
fn try_shift_fold(instrs: &[Instr], i: usize) -> Option<Vec<Instr>> {
    if i + 2 >= instrs.len() {
        return None;
    }

    let (rd, rs, mask) = match &instrs[i] {
        Instr::Andi(rd, rs, Imm::Hex(mask)) => (*rd, *rs, *mask),
        _ => return None,
    };

    let a = match &instrs[i + 1] {
        Instr::Srli(dst, src, a) if *dst == rd && *src == rd => *a,
        _ => return None,
    };

    let b = match &instrs[i + 2] {
        Instr::Slli(dst, src, b) if *dst == rd && *src == rd => *b,
        _ => return None,
    };

    let new_mask = mask & !((1u32 << a) - 1);
    let mut replacement = vec![Instr::Andi(rd, rs, Imm::Hex(new_mask))];

    match b.cmp(&a) {
        Ordering::Greater => replacement.push(Instr::Slli(rd, rd, b - a)),
        Ordering::Less    => replacement.push(Instr::Srli(rd, rd, a - b)),
        Ordering::Equal   => {}
    }

    Some(replacement)
}
