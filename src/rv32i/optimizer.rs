use std::cmp::Ordering;

use super::{Imm, Instr};

pub fn peephole(instrs: &mut Vec<Instr>) {
    let mut result = Vec::with_capacity(instrs.len());
    let mut i = 0;
    while i < instrs.len() {
        if let Some(folded) = try_shift_fold(instrs, i) {
            result.extend(folded);
            i += 3;
        } else {
            result.push(instrs[i].clone());
            i += 1;
        }
    }
    *instrs = result;
}

fn try_shift_fold(instrs: &[Instr], i: usize) -> Option<Vec<Instr>> {
    if i + 2 >= instrs.len() {
        return None;
    }

    let (rd, rs, mask) = match &instrs[i] {
        Instr::Andi(rd, rs, Imm::Hex(mask)) => (*rd, *rs, *mask),
        _ => return None,
    };

    let srli = match &instrs[i + 1] {
        Instr::Srli(dst, src, shamt) if *dst == rd && *src == rd => *shamt,
        _ => return None,
    };

    let slli = match &instrs[i + 2] {
        Instr::Slli(dst, src, shamt) if *dst == rd && *src == rd => *shamt,
        _ => return None,
    };

    if srli >= u32::BITS {
        return None;
    }

    let new_mask = mask & !((1u32 << srli) - 1);
    let mut replacement = vec![Instr::Andi(rd, rs, Imm::Hex(new_mask))];
    match slli.cmp(&srli) {
        Ordering::Greater => replacement.push(Instr::Slli(rd, rd, slli - srli)),
        Ordering::Less => replacement.push(Instr::Srli(rd, rd, srli - slli)),
        Ordering::Equal => {}
    }
    Some(replacement)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rv32i::Reg::*;

    #[test]
    fn folds_andi_srli_slli_to_net_left_shift() {
        let mut instrs = vec![
            Instr::Andi(T4, T0, Imm::Hex(0xf)),
            Instr::Srli(T4, T4, 2),
            Instr::Slli(T4, T4, 8),
        ];

        peephole(&mut instrs);

        assert_eq!(
            instrs,
            vec![Instr::Andi(T4, T0, Imm::Hex(0xc)), Instr::Slli(T4, T4, 6),]
        );
    }

    #[test]
    fn folds_andi_srli_slli_to_mask_only_when_shifts_cancel() {
        let mut instrs = vec![
            Instr::Andi(T4, T0, Imm::Hex(0xf)),
            Instr::Srli(T4, T4, 2),
            Instr::Slli(T4, T4, 2),
        ];

        peephole(&mut instrs);

        assert_eq!(instrs, vec![Instr::Andi(T4, T0, Imm::Hex(0xc))]);
    }
}
