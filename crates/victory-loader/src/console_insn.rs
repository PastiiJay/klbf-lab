//! Emulate x86-64 instructions the host CPU does not support.
//!
//! Based on XCL (github.com/ivall/xcl, GPL-2.0-only): `insn.rs` and `eh.rs`.

#[path = "console_insn_decode.rs"]
mod decode;

use decode::{ConsoleOnly, LoopCheck, ZeroLoop};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering::Relaxed};
use windows_sys::Win32::System::Diagnostics::Debug::CONTEXT;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_EXECUTE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_EXECUTE_WRITECOPY, PAGE_GUARD,
    PAGE_NOACCESS, PAGE_READWRITE, PAGE_WRITECOPY, VirtualQuery,
};

const CONTINUE_EXECUTION: i32 = -1;

/// Whether what Windows said about memory is remembered per thread (the regions it found
/// writable, the code bytes it found readable) and not asked again at every emulated
/// instruction. A query walks the pages of the region it answers for: asked from outside, it
/// took 2-9 ms in the title's heaps of 0.6-2.2 GB and 89 ms in its 6.2 GB page pool (run 30),
/// and the title zeroes blocks of them with `clzero` loops all through a frame: six of its
/// threads spent 25-78 % of their time inside that query (runs 29 and 30), about 27 s of
/// processor time in Windows per 9 s frame. On by default; `VICTORY_CLZERO_ASK=1` asks every
/// time, as before.
fn remembers() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLZERO_ASK").is_none_or(|v| v != "1"))
}

/// Answers a remembered region gives before Windows is asked about it again: one freed or
/// protected meanwhile is forgotten then.
const ANSWERS: u32 = 4096;

thread_local! {
    /// The regions Windows last said are committed and writable (start, end), newest first,
    /// and the answers they gave since they were emptied.
    static WRITABLE: Cell<([(usize, usize); 4], u32)> = const { Cell::new(([(0, 0); 4], 0)) };
    /// Faulting addresses and the code bytes found readable there, newest first.
    static CODE: Cell<[(usize, usize); 8]> = const { Cell::new([(0, 0); 8]) };
}

/// Instructions emulated, the times Windows was asked about memory for them, and the bytes
/// `clzero` loops zeroed, since the start.
static EMULATED: AtomicU64 = AtomicU64::new(0);
static ASKED: AtomicU64 = AtomicU64::new(0);
static ZEROED: AtomicU64 = AtomicU64::new(0);

fn committed(va: usize) -> Option<MEMORY_BASIC_INFORMATION> {
    // SAFETY: plain data; the query fills a valid out pointer of the right size.
    let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let found = unsafe { VirtualQuery(va as *const _, &mut m, size_of_val(&m)) } != 0;
    ASKED.fetch_add(1, Relaxed);
    (found && m.State == MEM_COMMIT && m.Protect & (PAGE_NOACCESS | PAGE_GUARD) == 0).then_some(m)
}

fn readable_len(va: usize, bytes: usize) -> usize {
    let end = va.saturating_add(bytes);
    let mut at = va;
    while at < end {
        let Some(m) = committed(at) else { break };
        let Some(to) = (m.BaseAddress as usize).checked_add(m.RegionSize) else { break };
        at = to.min(end);
    }
    at - va
}

fn writable(va: usize, bytes: usize) -> bool {
    let Some(end) = va.checked_add(bytes) else { return false };
    writable_to(va, end) == Some(end)
}

/// Where the committed, writable memory that `va` lies in ends, at most `end`; None if `va`
/// is not in such memory. A region found is remembered ([`remembers`]).
/// shortcut: a remembered region that was freed or protected since is written all the same
/// for up to [`ANSWERS`] answers: the write then faults inside the handler, which the guard
/// on view memory opens as it does any write, and which is a crash otherwise.
fn writable_to(va: usize, end: usize) -> Option<usize> {
    let remember = remembers();
    if remember {
        let (regions, answers) = WRITABLE.with(Cell::get);
        if let Some(&(_, to)) = regions.iter().find(|&&(from, to)| from <= va && va < to) {
            let next = if answers >= ANSWERS { ([(0, 0); 4], 0) } else { (regions, answers + 1) };
            WRITABLE.with(|w| w.set(next));
            return Some(to.min(end));
        }
    }
    let m = committed(va)?;
    if !matches!(m.Protect & 0xff, PAGE_READWRITE | PAGE_EXECUTE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY) {
        return None;
    }
    let from = m.BaseAddress as usize;
    let to = from.checked_add(m.RegionSize)?;
    if remember {
        WRITABLE.with(|w| {
            let (mut regions, answers) = w.get();
            regions.rotate_right(1);
            regions[0] = (from, to);
            w.set((regions, answers));
        });
    }
    Some(to.min(end))
}

// A one-byte patch preserves instruction boundaries and is atomic on x86-64. Only the
// page containing that byte needs to change protection, even for a split instruction.
unsafe fn patch_nop(rip: usize, code: &[u8], opcode: u8) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
    use windows_sys::Win32::System::Memory::VirtualProtect;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let Some(at) = code.iter().position(|&b| b == 0x0f).map(|i| i + 1) else { return false };
    let ptr = (rip + at) as *mut u8;
    // Serialize protection changes (including patches at different sites on the same page).
    static PATCH: Mutex<()> = Mutex::new(());
    let Ok(_lock) = PATCH.lock() else { return false };
    let Some(m) = committed(ptr as usize) else { return false };
    if !matches!(m.Protect & 0xff, PAGE_EXECUTE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE | PAGE_EXECUTE_WRITECOPY) {
        return false;
    }
    // SAFETY: the byte was read by the decoder and is still in a committed executable page.
    // A relaxed byte load is supported on read-only memory on x86-64.
    match unsafe { (&*ptr.cast::<AtomicU8>()).load(Relaxed) } {
        0x1f => return true, // another thread already handled the faulting instruction
        b if b == opcode => {}
        _ => return false,
    }
    let mut old = 0;
    // SAFETY: only the validated opcode byte is changed. Restore its page's protection
    // before releasing the lock, and flush the instruction cache before resuming execution.
    unsafe {
        if VirtualProtect(ptr.cast(), 1, PAGE_EXECUTE_READWRITE, &mut old) == 0 {
            return false;
        }
        let patched = AtomicU8::from_ptr(ptr).compare_exchange(opcode, 0x1f, Relaxed, Relaxed).is_ok();
        let restored = VirtualProtect(ptr.cast(), 1, old, &mut old) != 0;
        if patched {
            FlushInstructionCache(GetCurrentProcess(), ptr.cast(), 1);
        }
        patched && restored
    }
}

fn zero_line(addr: u64) -> bool {
    let line = addr as usize & !63;
    if !writable(line, 64) {
        return false;
    }
    // SAFETY: the whole line is committed and writable.
    unsafe { std::ptr::write_bytes(line as *mut u8, 0, 64) };
    true
}

// Preserve the existing bound on the amount of work performed in one exception.
const MAX_ZERO_LINES: u64 = (2 << 30) / 64 - 1;
const DEC_FLAGS: u32 = 0x8d4; // DEC updates the same status flags as CMP except for CF.

// Returns the number of bytes to advance RIP. Partial progress takes the original back edge.
fn run_zero_loop(ctx: &mut CONTEXT, loop_: ZeroLoop, len: usize, single_len: usize) -> Option<usize> {
    use crate::crash::{CMP_FLAGS, cmp_flags, gpr};

    let start = ctx.Rax;
    let count = match loop_.check {
        LoopCheck::Pointer { end_reg, .. } => (*gpr(ctx, end_reg)).saturating_sub(start).div_ceil(64).max(1),
        LoopCheck::Countdown { reg, wide } => {
            let n = *gpr(ctx, reg);
            if wide { n } else { u64::from(n as u32) }
        }
    };
    // A mismatched induction register, a wrapping countdown, or a very large loop is
    // stepped once. Its subsequent instructions and branch run natively.
    if !(1..=MAX_ZERO_LINES).contains(&count) || *gpr(ctx, loop_.base) != start {
        return zero_line(start).then_some(single_len);
    }
    let Some(finish) = start.checked_add(count * 64) else {
        return zero_line(start).then_some(single_len);
    };
    let from = start as usize & !63;
    let end = ((finish as usize - 64) & !63).checked_add(64)?;
    let mut at = from;
    while at < end {
        let Some(to) = writable_to(at, end) else { break };
        // SAFETY: the query (or cached answer) covers this contiguous writable span.
        unsafe { std::ptr::write_bytes(at as *mut u8, 0, to - at) };
        at = to;
    }
    let bytes = at - from;
    if bytes == 0 {
        return None;
    }
    ZEROED.fetch_add(bytes as u64, Relaxed);
    let next = start + bytes as u64;
    ctx.Rax = next;
    *gpr(ctx, loop_.base) = next;
    match loop_.check {
        LoopCheck::Pointer { end_reg, reversed } => {
            let limit = *gpr(ctx, end_reg);
            let flags = if reversed { cmp_flags(limit, next) } else { cmp_flags(next, limit) };
            ctx.EFlags = (ctx.EFlags & !CMP_FLAGS) | flags;
        }
        LoopCheck::Countdown { reg, wide } => {
            let result = count - (bytes / 64) as u64;
            let sign_bit = if wide { 63 } else { 31 };
            let flags = u32::from((result as u8).count_ones().is_multiple_of(2)) << 2
                | u32::from(result & 0xf == 0xf) << 4
                | u32::from(result == 0) << 6
                | u32::from(result >> sign_bit != 0) << 7
                | u32::from(result == (1u64 << sign_bit) - 1) << 11;
            // DEC preserves CF. A preceding ADD of 64 cannot carry on this checked span.
            let mask = if loop_.add { CMP_FLAGS } else { DEC_FLAGS };
            ctx.EFlags = (ctx.EFlags & !mask) | flags;
            *gpr(ctx, reg) = result;
        }
    }
    Some(if at == end { len } else { loop_.target })
}

/// Emulate a supported illegal instruction at `ctx.Rip`, or return 0 to continue dispatch.
///
/// # Safety
/// `ctx` is the faulting thread's context, from the exception being dispatched.
pub(crate) unsafe fn emulate(ctx: &mut CONTEXT) -> i32 {
    let rip = ctx.Rip as usize;
    // The same few instructions fault again and again; cache the readable length, not the bytes.
    let mut bytes = [0; 32];
    let known = if remembers() { CODE.with(Cell::get) } else { [(0, 0); 8] };
    let n = match known.iter().find(|&&(at, n)| at == rip && n > 0) {
        Some(&(_, n)) => n,
        None => {
            let n = readable_len(rip, bytes.len());
            // A split instruction is rechecked if the adjacent page changes protection.
            if remembers() && n == bytes.len() && rip & 0xfff <= 0x1000 - bytes.len() {
                let mut known = known;
                known.rotate_right(1);
                known[0] = (rip, n);
                CODE.with(|c| c.set(known));
            }
            n
        }
    };
    for (i, byte) in bytes[..n].iter_mut().enumerate() {
        // SAFETY: the byte is readable. An atomic load matches the patcher's atomic write.
        *byte = unsafe { (&*((rip + i) as *const AtomicU8)).load(Relaxed) };
    }
    // SAFETY: this slice is the snapshot at the live faulting address in `ctx`.
    unsafe { emulate_slice(ctx, &bytes[..n]) }.map_or(0, |_| CONTINUE_EXECUTION)
}

// Returns the number of bytes advanced, including zero when resuming a partially emulated loop.
unsafe fn emulate_slice(ctx: &mut CONTEXT, code: &[u8]) -> Option<usize> {
    let rip = ctx.Rip as usize;
    let (what, mut len) = decode::decode(code)?;
    static SEEN: Mutex<Vec<std::mem::Discriminant<ConsoleOnly>>> = Mutex::new(Vec::new());
    if !matches!(what, ConsoleOnly::Ud2 | ConsoleOnly::PatchedInsn)
        && let Ok(mut seen) = SEEN.lock()
        && !seen.contains(&std::mem::discriminant(&what))
    {
        seen.push(std::mem::discriminant(&what));
        tracing::warn!(instruction = ?what, at = format_args!("{rip:#x}"), "unsupported CPU instruction emulated (logged once)");
    }
    match what {
        ConsoleOnly::Ud2 => return None,
        ConsoleOnly::PatchedInsn => {} // the thread faulted before another patched the opcode
        ConsoleOnly::Clwb | ConsoleOnly::Monitorx | ConsoleOnly::Mwaitx => {
            let opcode = if what == ConsoleOnly::Clwb { 0xae } else { 0x01 };
            // SAFETY: these are the decoded bytes at RIP. Only the second opcode byte changes.
            let patched = unsafe { patch_nop(rip, &code[..len], opcode) };
            if !patched && what == ConsoleOnly::Mwaitx {
                std::hint::spin_loop();
            }
        }
        ConsoleOnly::Clzero => {
            // With an address-size prefix (0x67) the line is at EAX, zero-extended.
            let a32 = code[..len - 3].contains(&0x67);
            let addr = if a32 { u64::from(ctx.Rax as u32) } else { ctx.Rax };
            if !zero_line(addr) {
                return None;
            }
        }
        ConsoleOnly::ClzeroLoop(loop_) => {
            let single_len = code.iter().position(|&b| b == 0x0f)? + 3;
            len = run_zero_loop(ctx, loop_, len, single_len)?;
        }
        ConsoleOnly::Rdpru => {
            let mut v = 0i64;
            // SAFETY: a valid out pointer.
            unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut v) };
            let v = if ctx.Rcx as u32 <= 1 { v as u64 } else { 0 };
            ctx.Rax = v & 0xFFFF_FFFF;
            ctx.Rdx = v >> 32;
        }
        ConsoleOnly::Rdpid(r) => {
            // SAFETY: no preconditions.
            let cpu = unsafe { windows_sys::Win32::System::Threading::GetCurrentProcessorNumber() } as u64;
            let regs: [&mut u64; 16] = [
                &mut ctx.Rax, &mut ctx.Rcx, &mut ctx.Rdx, &mut ctx.Rbx, &mut ctx.Rsp, &mut ctx.Rbp, &mut ctx.Rsi, &mut ctx.Rdi, &mut ctx.R8, &mut ctx.R9, &mut ctx.R10, &mut ctx.R11, &mut ctx.R12, &mut ctx.R13,
                &mut ctx.R14, &mut ctx.R15,
            ];
            *regs[r as usize & 15] = cpu;
        }
        // the low quadword changes; the high one is zeroed, as a Zen 5 does
        ConsoleOnly::Extrq { dst, src, len, idx } => {
            // SAFETY: the context's XMM registers (CONTEXT_FLOATING_POINT is captured for exceptions).
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let (len, idx) = match src {
                Some(s) => {
                    let c = x[s as usize & 15].Low;
                    (c as u8, (c >> 8) as u8)
                }
                None => (len, idx),
            };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::extrq(d.Low, len, idx);
            d.High = 0;
        }
        ConsoleOnly::Insertq { dst, src, reg, len, idx } => {
            // SAFETY: as above.
            let x = unsafe { &mut ctx.Anonymous.FltSave.XmmRegisters };
            let s = x[src as usize & 15];
            let (len, idx) = if reg { (s.High as u8, (s.High >> 8) as u8) } else { (len, idx) };
            let d = &mut x[dst as usize & 15];
            d.Low = decode::insertq(d.Low, s.Low, len, idx);
            d.High = 0;
        }
    }
    ctx.Rip += len as u64;
    // Each live invocation is an exception: a trip into Windows and back.
    let emulated = EMULATED.fetch_add(1, Relaxed) + 1;
    if emulated >= 1 << 16 && emulated.is_power_of_two() {
        tracing::info!(
            emulated,
            asked = ASKED.load(Relaxed),
            zeroed_mb = ZEROED.load(Relaxed) >> 20,
            "CPU instructions emulated, memory queries, and bytes zeroed by clzero loops"
        );
    }
    Some(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Memory::{
        CreateFileMappingW, FILE_MAP_COPY, FILE_MAP_WRITE, MEM_RELEASE, MEM_RESERVE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
        UnmapViewOfFile, VirtualAlloc, VirtualFree, VirtualProtect,
    };

    #[test]
    fn patches_wait_and_clwb_in_place() {
        // SAFETY: our private allocation holds the test instructions and a return. Only the
        // patched sequence is executed, so neither MONITORX nor CLWB support is needed.
        unsafe {
            let p = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!p.is_null());
            let start = p.add(0xfff);
            let code = [0x0f, 0x01, 0xfa, 0x67, 0x0f, 0x01, 0xfb, 0x66, 0x41, 0x0f, 0xae, 0x74, 0x24, 0x08, 0xc3];
            std::ptr::copy_nonoverlapping(code.as_ptr(), start, code.len());
            let mut old = 0;
            assert_ne!(VirtualProtect(p.cast(), 0x1000, PAGE_EXECUTE_READWRITE, &mut old), 0);
            assert_ne!(VirtualProtect(p.add(0x1000).cast(), 0x1000, PAGE_EXECUTE_READ, &mut old), 0);
            for (offset, len) in [(0, 3), (3, 4), (7, 7)] {
                // A second dispatch simulates a fault queued before the opcode was patched.
                for _ in 0..2 {
                    let mut ctx: CONTEXT = std::mem::zeroed();
                    ctx.Rip = start.add(offset) as u64;
                    assert_eq!(emulate(&mut ctx), CONTINUE_EXECUTION);
                    assert_eq!(ctx.Rip, start.add(offset + len) as u64);
                }
            }
            let mut expected = code;
            for offset in [1, 5, 10] {
                expected[offset] = 0x1f;
            }
            assert_eq!(std::slice::from_raw_parts(start, code.len()), &expected);
            assert_eq!(committed(p as usize).unwrap().Protect, PAGE_EXECUTE_READWRITE);
            assert_eq!(committed(p.add(0x1000) as usize).unwrap().Protect, PAGE_EXECUTE_READ);
            let f: unsafe extern "win64" fn() = std::mem::transmute(start);
            f();
            assert_ne!(VirtualFree(p.cast(), 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn clzero_accelerates_pointer_and_countdown_loops() {
        #[repr(align(64))]
        struct Aligned([u8; 640]);
        let mut arena = Aligned([0xa5; 640]);
        let pointer = [0x0f, 0x01, 0xfc, 0x49, 0x83, 0xc0, 0x40, 0x4c, 0x89, 0xc0, 0x4d, 0x39, 0xc8, 0x72, 0xf1];
        let countdown = [0x0f, 0x01, 0xfc, 0xff, 0xc9, 0x48, 0x8d, 0x40, 0x40, 0x75, 0xf5];
        let reversed = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xc2, 0x77, 0xf4];
        // SAFETY: all targets are in the aligned arena; these calls emulate without executing the instructions.
        unsafe {
            let start = arena.0.as_mut_ptr().add(64) as u64;
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rax = start;
            ctx.R8 = start;
            ctx.R9 = start + 7 * 64 + 16;
            assert_eq!(emulate_slice(&mut ctx, &pointer), Some(15));
            assert_eq!(ctx.Rax, start + 8 * 64);
            assert_eq!(ctx.R8, ctx.Rax);
            assert!(arena.0[..64].iter().chain(&arena.0[9 * 64..]).all(|&b| b == 0xa5));
            assert!(arena.0[64..9 * 64].iter().all(|&b| b == 0));

            arena.0.fill(0xa5);
            ctx.Rax = start + 5;
            ctx.Rcx = 0xffff_ffff_0000_0005;
            ctx.EFlags = 1;
            assert_eq!(emulate_slice(&mut ctx, &countdown), Some(11));
            assert_eq!(ctx.Rax, start + 5 + 5 * 64);
            assert_eq!(ctx.Rcx, 0);
            assert_eq!(ctx.EFlags & crate::crash::CMP_FLAGS, 0x45); // CF preserved; PF and ZF set
            assert!(arena.0[..64].iter().chain(&arena.0[6 * 64..]).all(|&b| b == 0xa5));
            assert!(arena.0[64..6 * 64].iter().all(|&b| b == 0));

            ctx.Rax = start;
            ctx.Rdx = start + 64;
            ctx.EFlags = 0x200 | crate::crash::CMP_FLAGS;
            assert_eq!(emulate_slice(&mut ctx, &reversed), Some(12));
            assert_eq!(ctx.Rax, ctx.Rdx);
            assert_eq!(ctx.EFlags, 0x244); // CMP with equal operands sets PF and ZF, leaving IF alone
        }
    }

    #[test]
    fn clzero_handles_writecopy_guard_and_overflow() {
        WRITABLE.with(|w| w.set(([(0, 0); 4], 0)));
        // SAFETY: both views and the guarded allocation are private to this test and freed below.
        unsafe {
            let mapping = CreateFileMappingW(INVALID_HANDLE_VALUE, std::ptr::null(), PAGE_READWRITE, 0, 0x1000, std::ptr::null());
            assert!(!mapping.is_null());
            let shared = MapViewOfFile(mapping, FILE_MAP_WRITE, 0, 0, 0x1000);
            let copy = MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, 0x1000);
            let single_copy = MapViewOfFile(mapping, FILE_MAP_COPY, 0, 0, 0x1000);
            assert!(!shared.Value.is_null() && !copy.Value.is_null() && !single_copy.Value.is_null());
            let wc = copy.Value.cast::<u8>();
            std::ptr::write_bytes(shared.Value.cast::<u8>(), 0xa5, 0x1000);
            assert_eq!(committed(wc as usize).unwrap().Protect & 0xff, PAGE_WRITECOPY);

            let single = [0x0f, 0x01, 0xfc];
            let loop_code = [0x0f, 0x01, 0xfc, 0x48, 0x83, 0xc0, 0x40, 0x48, 0x39, 0xd0, 0x72, 0xf4];
            let mut ctx: CONTEXT = std::mem::zeroed();
            ctx.Rax = single_copy.Value as u64;
            assert_eq!(emulate_slice(&mut ctx, &single), Some(3));
            let line = std::slice::from_raw_parts(single_copy.Value.cast::<u8>(), 128);
            assert!(line[..64].iter().all(|&b| b == 0));
            assert!(line[64..].iter().all(|&b| b == 0xa5));

            ctx.Rax = wc as u64;
            ctx.Rdx = wc.add(0x400) as u64;
            assert_eq!(emulate_slice(&mut ctx, &loop_code), Some(12));
            assert!(std::slice::from_raw_parts(wc, 0x400).iter().all(|&b| b == 0));
            assert!(std::slice::from_raw_parts(wc.add(0x400), 64).iter().all(|&b| b == 0xa5));
            assert_eq!(shared.Value.cast::<u8>().read(), 0xa5);

            let pages = VirtualAlloc(std::ptr::null(), 0x2000, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE).cast::<u8>();
            assert!(!pages.is_null());
            let guard = pages.add(0x1000);
            let last_line = guard.sub(64);
            std::ptr::write_bytes(last_line, 0xa5, 64);
            let mut old = 0;
            assert_ne!(VirtualProtect(guard.cast(), 0x1000, PAGE_READWRITE | PAGE_GUARD, &mut old), 0);
            ctx.Rax = last_line as u64;
            ctx.Rdx = guard.add(64) as u64;
            ctx.Rip = loop_code.as_ptr() as u64;
            assert_eq!(emulate_slice(&mut ctx, &loop_code), Some(0)); // original back edge
            assert_eq!(ctx.Rax, guard as u64);
            assert_eq!(ctx.Rip, loop_code.as_ptr() as u64);
            assert!(std::slice::from_raw_parts(last_line, 64).iter().all(|&b| b == 0));
            assert_eq!(emulate_slice(&mut ctx, &loop_code), None);
            for bad in [guard as u64, u64::MAX, u64::MAX - 31] {
                ctx.Rax = bad;
                assert_eq!(emulate_slice(&mut ctx, &single), None);
            }
            assert_ne!(VirtualFree(pages.cast(), 0, MEM_RELEASE), 0);
            for view in [single_copy, copy, shared] {
                assert_ne!(UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view.Value }), 0);
            }
            assert_ne!(CloseHandle(mapping), 0);
        }
    }
}
