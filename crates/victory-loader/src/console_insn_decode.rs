//! Decode x86-64 instructions handled by the illegal-instruction emulator.

/// An instruction handled at the faulting address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConsoleOnly {
    /// `clwb m8`: write a cache line back – nothing to do, the PC's memory is coherent
    Clwb,
    /// `clzero`: zero the 64-byte cache line at rAX
    Clzero,
    /// A `clzero` loop with a 64-byte stride and a pointer bound or countdown.
    ClzeroLoop(ZeroLoop),
    /// An instruction changed into a same-length NOP by another thread.
    PatchedInsn,
    /// `rdpru`: ecx = 0 MPERF / 1 APERF → edx:eax (a counter)
    Rdpru,
    /// `monitorx` / `mwaitx`: a wait the emulation skips
    Monitorx,
    Mwaitx,
    /// `rdpid r64`: TSC_AUX (the processor number) into the register (CONTEXT order: rax, rcx,
    /// rdx, rbx, rsp, rbp, rsi, rdi, r8..r15)
    Rdpid(u8),
    /// SSE4a `extrq xmm, imm8 (length), imm8 (index)` / `extrq xmm1, xmm2` (`src` Some: the
    /// length and index in xmm2's bits 5:0 and 13:8): a bit field of the low quadword, moved
    /// down and zero-extended (the high quadword zeroed). Registers are XMM numbers 0..15.
    Extrq { dst: u8, src: Option<u8>, len: u8, idx: u8 },
    /// SSE4a `insertq xmm1, xmm2, imm8 (length), imm8 (index)` / `insertq xmm1, xmm2` (`reg`:
    /// the length and index in xmm2's bits 69:64 and 77:72): the low bits of xmm2 put into a
    /// bit field of xmm1's low quadword.
    Insertq { dst: u8, src: u8, reg: bool, len: u8, idx: u8 },
    /// `ud2`: the title's own trap (not emulated)
    Ud2,
}

/// A field length of SSE4a's six bits: 0 is 64.
fn field_len(len: u8) -> u32 {
    match len & 63 {
        0 => 64,
        n => n as u32,
    }
}

/// EXTRQ on a low quadword: `len` bits from bit `idx`, zero-extended (a field past bit 63 is
/// undefined on the console; here it is what a shift gives).
pub fn extrq(v: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let field = v.checked_shr(idx).unwrap_or(0);
    if len >= 64 {
        field
    } else {
        field & ((1u64 << len) - 1)
    }
}

/// INSERTQ on a low quadword: the low `len` bits of `src` put at bit `idx` of `dst`.
pub fn insertq(dst: u64, src: u64, len: u8, idx: u8) -> u64 {
    let (len, idx) = (field_len(len), (idx & 63) as u32);
    let ones = if len >= 64 { u64::MAX } else { (1u64 << len) - 1 };
    let mask = ones.checked_shl(idx).unwrap_or(0);
    (dst & !mask) | ((src & ones).checked_shl(idx).unwrap_or(0) & mask)
}

/// The length of a ModRM-encoded operand (ModRM, SIB and displacement) at `b[0]`, None when the
/// bytes run out.
fn modrm_len(b: &[u8]) -> Option<usize> {
    let modrm = *b.first()?;
    let (md, rm) = (modrm >> 6, modrm & 7);
    if md == 3 {
        return Some(1);
    }
    let mut len = 1;
    let mut base = rm;
    if rm == 4 {
        base = *b.get(1)? & 7;
        len += 1;
    }
    len += match md {
        0 if base == 5 => 4, // disp32 (rip-relative, or a SIB without a base)
        0 => 0,
        1 => 1,
        _ => 4,
    };
    (b.len() >= len).then_some(len)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZeroLoop {
    pub base: u8,
    pub add: bool,
    pub check: LoopCheck,
    /// Offset of the back-edge target from the faulting instruction pointer.
    pub target: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopCheck {
    Pointer { end_reg: u8, reversed: bool },
    Countdown { reg: u8, wide: bool },
}

// ADD base, 64 or LEA base, [base + 64]. LEA leaves the incoming flags intact.
fn advance(code: &[u8]) -> Option<(u8, bool, usize)> {
    let &[rex @ 0x48..=0x4f, op, m, ..] = code else { return None };
    let base = (m & 7) | (rex & 1) << 3;
    match op {
        0x83 if m & 0xf8 == 0xc0 && code.get(3) == Some(&64) => Some((base, true, 4)),
        0x8d if m >> 6 == 1 && ((m >> 3) & 7 | (rex & 4) << 1) == base => {
            let disp = if m & 7 == 4 {
                // An RSP/R12 base needs SIB, with no index.
                if rex & 2 != 0 || code.get(3) != Some(&0x24) {
                    return None;
                }
                4
            } else {
                3
            };
            (*code.get(disp)? == 64).then_some((base, false, disp + 1))
        }
        _ => None,
    }
}

// DEC r32/r64; not INC, a memory operand, or a 16-bit counter.
fn countdown(code: &[u8]) -> Option<(u8, bool, usize)> {
    let (rex, b) = match code {
        &[rex @ 0x40..=0x4f, ref b @ ..] => (rex, b),
        _ => (0, code),
    };
    let &[0xff, m @ 0xc8..=0xcf, ..] = b else { return None };
    Some(((m & 7) | (rex & 1) << 3, rex & 8 != 0, 2 + usize::from(rex != 0)))
}

// MOV rax, base: a non-RAX induction register must be copied back before the branch.
fn copies_to_rax(code: &[u8], base: u8) -> bool {
    code.starts_with(&[0x48 | ((base >> 3) << 2), 0x89, 0xc0 | ((base & 7) << 3)])
        || code.starts_with(&[0x48 | (base >> 3), 0x8b, 0xc0 | (base & 7)])
}

fn zero_loop(code: &[u8], opcode: usize) -> Option<(ZeroLoop, usize)> {
    let body = opcode + 3;
    let backward = |at: usize, cc: u8| {
        if *code.get(at)? != cc {
            return None;
        }
        let len = at + 2;
        let dest = len.checked_add_signed(*code.get(at + 1)? as i8 as isize)?;
        // A back edge may target the original instruction or skip its ignored prefixes.
        (dest == 0 || dest == opcode).then_some((dest, len))
    };

    let first_dec = countdown(&code[body..]);
    let step = body + first_dec.map_or(0, |(_, _, len)| len);
    let (base, add, step_len) = advance(&code[step..])?;
    let mut at = step + step_len;
    if let Some((reg, wide, n)) = first_dec.or_else(|| countdown(&code[at..])) {
        // ADD after DEC would make JNE depend on the pointer instead of the counter.
        if base != 0 || reg == 0 || (first_dec.is_some() && add) {
            return None;
        }
        if first_dec.is_none() {
            at += n;
        }
        let (target, len) = backward(at, 0x75)?;
        return Some((ZeroLoop { base, add, check: LoopCheck::Countdown { reg, wide }, target }, len));
    }

    let copy_before = base != 0 && copies_to_rax(&code[at..], base);
    if copy_before {
        at += 3;
    }
    // CMP r64, r64: both encodings are allowed. For a reversed comparison
    // the corresponding unsigned back edge is JA instead of JB.
    let &[rex @ 0x48..=0x4f, op @ (0x39 | 0x3b), m @ 0xc0..=0xff] = code.get(at..at + 3)? else { return None };
    let rm = (m & 7) | (rex & 1) << 3;
    let reg = ((m >> 3) & 7) | (rex & 4) << 1;
    let (lhs, rhs) = if op == 0x39 { (rm, reg) } else { (reg, rm) };
    let is_pointer = |reg| reg == base || (copy_before && reg == 0);
    let (end_reg, reversed) = if is_pointer(lhs) { (rhs, false) } else if is_pointer(rhs) { (lhs, true) } else { return None };
    // Neither the induction register nor RAX can be a fixed bound.
    if end_reg == base || end_reg == 0 {
        return None;
    }
    at += 3;
    if base != 0 && !copy_before {
        if !copies_to_rax(&code[at..], base) {
            return None;
        }
        at += 3;
    }
    let (target, len) = backward(at, if reversed { 0x77 } else { 0x72 })?;
    Some((ZeroLoop { base, add, check: LoopCheck::Pointer { end_reg, reversed }, target }, len))
}

/// The handled instruction at the start of `code` and its length.
pub fn decode(lookahead: &[u8]) -> Option<(ConsoleOnly, usize)> {
    // Only the loop matcher can look past the maximum x86 instruction length.
    let code = &lookahead[..lookahead.len().min(15)];
    let mut i = 0;
    // `rep`: the last of F2 / F3, which decides between them (and over 66) for SSE4a
    let (mut p66, mut pf3, mut rep, mut rex) = (false, false, 0u8, 0u8);
    while let Some(&b) = code.get(i) {
        match b {
            0x66 => p66 = true,
            0xF2 => rep = 0xF2,
            0xF3 => {
                pf3 = true;
                rep = 0xF3;
            }
            0x2E | 0x3E | 0x26 | 0x36 | 0x64 | 0x65 | 0x67 => {}
            // REX counts only right before the opcode: one before a legacy prefix is ignored
            0x40..=0x4F => {
                rex = b;
                i += 1;
                continue;
            }
            _ => break,
        }
        rex = 0;
        i += 1;
    }
    if *code.get(i)? != 0x0F {
        return None;
    }
    match (*code.get(i + 1)?, code.get(i + 2).copied()) {
        (0x01, Some(0xFC)) => {
            // An address-size prefix takes the address from EAX, not RAX.
            if !code[..i].contains(&0x67)
                && let Some((loop_, len)) = zero_loop(lookahead, i)
            {
                return Some((ConsoleOnly::ClzeroLoop(loop_), len));
            }
            Some((ConsoleOnly::Clzero, i + 3))
        }
        (0x01, Some(0xFD)) => Some((ConsoleOnly::Rdpru, i + 3)),
        (0x01, Some(0xFA)) => Some((ConsoleOnly::Monitorx, i + 3)),
        (0x01, Some(0xFB)) => Some((ConsoleOnly::Mwaitx, i + 3)),
        // The second opcode byte is the only one changed by the patcher.
        (0x1F, Some(0xFA | 0xFB)) => Some((ConsoleOnly::PatchedInsn, i + 3)),
        (0x1F, Some(m)) if p66 && rep == 0 && (m >> 3) & 7 == 6 && m >> 6 != 3 => {
            Some((ConsoleOnly::PatchedInsn, i + 2 + modrm_len(&code[i + 2..])?))
        }
        (0x0B, _) => Some((ConsoleOnly::Ud2, i + 2)),
        // SSE4a: register operands only (ModRM mod 3); REX.R extends reg, REX.B rm
        // (the last of F2 / F3 decides: F2 INSERTQ, F3 #UD; neither with 66: EXTRQ)
        (0x78 | 0x79, Some(m)) if (rep == 0xF2 || (rep == 0 && p66)) && m >> 6 == 3 => {
            let reg = ((m >> 3) & 7) | (rex & 4) << 1;
            let rm = (m & 7) | (rex & 1) << 3;
            let imm = |k: usize| code.get(i + 3 + k).copied();
            match (code[i + 1], rep == 0xF2) {
                // extrq xmm(rm), imm8, imm8 – the ModRM reg field is /0
                (0x78, false) if (m >> 3) & 7 == 0 => Some((ConsoleOnly::Extrq { dst: rm, src: None, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, false) => Some((ConsoleOnly::Extrq { dst: reg, src: Some(rm), len: 0, idx: 0 }, i + 3)),
                (0x78, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: false, len: imm(0)?, idx: imm(1)? }, i + 5)),
                (0x79, true) => Some((ConsoleOnly::Insertq { dst: reg, src: rm, reg: true, len: 0, idx: 0 }, i + 3)),
                _ => None,
            }
        }
        // 66 0F AE /6 with a memory operand (without 66 it is XSAVEOPT; F2/F3 are invalid).
        (0xAE, Some(m)) if p66 && rep == 0 && (m >> 3) & 7 == 6 && m >> 6 != 3 => Some((ConsoleOnly::Clwb, i + 2 + modrm_len(&code[i + 2..])?)),
        // F3 0F C7 /7 with a register operand
        (0xC7, Some(m)) if pf3 && (m >> 3) & 7 == 7 && m >> 6 == 3 => Some((ConsoleOnly::Rdpid((m & 7) | (rex & 1) << 3), i + 3)),
        _ => None,
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// EXTRQ / INSERTQ as AMD's manual gives them: a length of 0 is 64.
    #[test]
    fn sse4a_bit_fields() {
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 8, 16), 0xBC);
        assert_eq!(extrq(0x1234_5678_9ABC_DEF0, 0, 0), 0x1234_5678_9ABC_DEF0);
        assert_eq!(extrq(0x8000_0000_0000_0000, 1, 63), 1);
        assert_eq!(insertq(0xFFFF_FFFF_FFFF_FFFF, 0xAB, 8, 8), 0xFFFF_FFFF_FFFF_ABFF);
        assert_eq!(insertq(0, 0x1FF, 4, 60), 0xF000_0000_0000_0000);
        assert_eq!(insertq(0x1111, 0xDEAD_BEEF_0000_0000, 0, 0), 0xDEAD_BEEF_0000_0000);
        for (v, len, idx) in [(0x0123_4567_89AB_CDEFu64, 12u8, 20u8), (u64::MAX, 63, 1), (0x55, 3, 0)] {
            assert_eq!(extrq(insertq(0, v, len, idx), len, idx), extrq(v, len, 0));
        }
    }

    #[test]
    fn decodes_the_console_only_instructions() {
        assert_eq!(decode(&[0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 3)));
        // `cmp rax, r10` / `cmp rax, rdx` differ only in REX.R.
        for (rex, end_reg) in [(0x4c, 10), (0x48, 2)] {
            let check = LoopCheck::Pointer { end_reg, reversed: false };
            assert_eq!(
                decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, rex, 0x39, 0xD0, 0x72, 0xF4]),
                Some((ConsoleOnly::ClzeroLoop(ZeroLoop { base: 0, add: true, check, target: 0 }), 12))
            );
        }
        // the same loop after an address-size prefix: that one CLZERO alone
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, 0xD0, 0x72, 0xF4]), Some((ConsoleOnly::Clzero, 4)));
        // cmp rcx, rdx (48 39 D1), cmp [rax], rdx (48 39 10), cmp rax, rax (48 39 C0): NOT a clzero loop
        for bad_modrm in [0xD1u8, 0x10, 0xC0] {
            assert_eq!(
                decode(&[0x0F, 0x01, 0xFC, 0x48, 0x83, 0xC0, 0x40, 0x48, 0x39, bad_modrm, 0x72, 0xF4]),
                Some((ConsoleOnly::Clzero, 3)),
                "modrm {bad_modrm:#04x} must not decode as ClzeroLoop"
            );
        }
        assert_eq!(decode(&[0x0F, 0x01, 0xFD]), Some((ConsoleOnly::Rdpru, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFA]), Some((ConsoleOnly::Monitorx, 3)));
        assert_eq!(decode(&[0x0F, 0x01, 0xFB]), Some((ConsoleOnly::Mwaitx, 3)));
        assert_eq!(decode(&[0x0F, 0x0B]), Some((ConsoleOnly::Ud2, 2)));
        // clwb [rax]; clwb [rcx+0x40]; clwb [rsp+8]; clwb [rip+disp32]; clwb [r8]
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x71, 0x40]), Some((ConsoleOnly::Clwb, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24, 0x08]), Some((ConsoleOnly::Clwb, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x35, 1, 2, 3, 4]), Some((ConsoleOnly::Clwb, 8)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0xAE, 0x30]), Some((ConsoleOnly::Clwb, 5)));
        // rdpid rax; rdpid r9
        assert_eq!(decode(&[0xF3, 0x0F, 0xC7, 0xF8]), Some((ConsoleOnly::Rdpid(0), 4)));
        assert_eq!(decode(&[0xF3, 0x41, 0x0F, 0xC7, 0xF9]), Some((ConsoleOnly::Rdpid(9), 5)));
        // not console-only: clflushopt (/7), xsaveopt (no 66), a plain mov, truncated bytes
        // SSE4a, as a retail game has it (#216: insertq xmm0, xmm1 with length / index bytes)
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08, 0x10]), Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 8, idx: 16 }, 6)));
        assert_eq!(decode(&[0xF2, 0x45, 0x0F, 0x79, 0xC1]), Some((ConsoleOnly::Insertq { dst: 8, src: 9, reg: true, len: 0, idx: 0 }, 5)));
        assert_eq!(decode(&[0x66, 0x0F, 0x78, 0xC2, 0x04, 0x08]), Some((ConsoleOnly::Extrq { dst: 2, src: None, len: 4, idx: 8 }, 6)));
        assert_eq!(decode(&[0x66, 0x0F, 0x79, 0xCA]), Some((ConsoleOnly::Extrq { dst: 1, src: Some(2), len: 0, idx: 0 }, 4)));
        assert_eq!(decode(&[0xF2, 0x0F, 0x78, 0xC1, 0x08]), None, "an immediate missing");
        assert_eq!(decode(&[0x0F, 0x78, 0xC1, 0x08, 0x10]), None, "VMREAD without a prefix is not SSE4a");
        // prefixes as a Zen 5 runs them (Kalmarv on #209): the last of F2 / F3 decides, 66 next
        // to them does not; 0x67 and a REX before a legacy prefix are ignored
        let insertq = Some((ConsoleOnly::Insertq { dst: 0, src: 1, reg: false, len: 16, idx: 16 }, 7));
        for code in [[0x66, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF2, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x67, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x41, 0xF2, 0x0F, 0x78, 0xC1, 0x10, 0x10]] {
            assert_eq!(decode(&code), insertq, "{code:02x?}");
        }
        for code in [[0xF2, 0xF3, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0xF3, 0x66, 0x0F, 0x78, 0xC1, 0x10, 0x10], [0x66, 0xF3, 0x0F, 0x78, 0xC0, 0x10, 0x10]] {
            assert_eq!(decode(&code), None, "#UD on the CPU: {code:02x?}");
        }
        assert_eq!(decode(&[0x67, 0x0F, 0x01, 0xFC]), Some((ConsoleOnly::Clzero, 4)));
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x38]), None);
        assert_eq!(decode(&[0x0F, 0xAE, 0x30]), None);
        assert_eq!(decode(&[0x67, 0x0F, 0x1F, 0xFB]), Some((ConsoleOnly::PatchedInsn, 4)));
        assert_eq!(decode(&[0x66, 0x41, 0x0F, 0x1F, 0x74, 0x24, 0x08]), Some((ConsoleOnly::PatchedInsn, 7)));
        for op in [0xAE, 0x1F] {
            for rep in [0xF2, 0xF3] {
                assert_eq!(decode(&[0x66, rep, 0x0F, op, 0x30]), None);
                assert_eq!(decode(&[rep, 0x66, 0x0F, op, 0x30]), None);
            }
        }
        assert_eq!(decode(&[0x48, 0x8B, 0x01]), None);
        assert_eq!(decode(&[0x66, 0x0F, 0xAE, 0x74, 0x24]), None);
        assert_eq!(decode(&[0x0F]), None);
    }
}
