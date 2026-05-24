use super::{Instr, Reg};

pub(super) fn peephole(instrs: &mut Vec<Instr>) {
    let mut result = Vec::with_capacity(instrs.len());
    let mut i = 0;
    while i < instrs.len() {
        if i + 2 < instrs.len() {
            // look at instrs[i], instrs[i+1], instrs[i+2]
            // if pattern matches: push replacement, i += 3, continue
            match (&instrs[i + 1], &instrs[i + 2]) {
                (Instr::Srli(rreg, _, imm1), Instr::Slli(lreg, _, imm2)) => {
                    if *imm1 < *imm2 && *rreg == *lreg {
                        let _some_new_instruction = change_previous(&instrs[i], rreg, imm1);
                    }
                }
                _ => {}
            }
        }
        result.push(instrs[i].clone());
        i += 1;
    }
    *instrs = result;
}

fn change_previous(instr: &Instr, reg: &Reg, ignore: &u32) -> Option<Instr> {
    match instr {
        Instr::Addi(ireg, ireg2, imm) => {
            if *ireg == *reg {
                let mask = !0 << ignore;
                return Some(Instr::Addi(*ireg, *ireg2, imm & mask));
            }
        }
        _ => {}
    }

    return None;
}
