//! Settings for machines below the tested one, chosen when the user does
//! not choose: the Series S profile on a PC with little memory, and an
//! upscaler that is there.
//!
//! The Series S profile renders the world at 1280x720 and asks for the
//! title's upscaler every frame; on the tested machine it is about a third
//! faster than Series X and commits less memory, which is what decides
//! whether a 16 GB PC holds a run at all. Its upscale goes to AMD's
//! FidelityFX DLLs (`VICTORY_FFX_DIR`), and without them the title stops
//! once it upscales; the bilinear stand-in (`VICTORY_FFX_SIMPLE=1`) needs
//! no DLL.

use std::path::Path;

use crate::Console;

/// Below this much physical memory the Series S profile is the default.
/// A Series X run commits about 20 GB; on 16 GB PCs it ran out where the
/// Series S profile peaked at about 18 GB and held with the page file.
pub const SERIES_S_BELOW: u64 = 24 << 30;

/// The DLL `VICTORY_FFX_DIR` must hold for AMD's upscaler.
pub const FFX_LOADER: &str = "amd_fidelityfx_loader_dx12.dll";

/// The console the title is told it runs on, and why: the user's choice,
/// else Series S below [`SERIES_S_BELOW`] of physical memory, else Series X
/// (also when the memory is not known).
pub fn console(chosen: Option<Console>, physical: Option<u64>) -> (Console, &'static str) {
    match (chosen, physical) {
        (Some(c), _) => (c, "chosen with --console"),
        (None, Some(bytes)) if bytes < SERIES_S_BELOW => (
            Console::SeriesS,
            "less than 24 GB of memory: the Series S profile (1280x720, upscaled) commits less",
        ),
        (None, Some(_)) => (Console::SeriesX, "24 GB of memory or more"),
        (None, None) => (Console::SeriesX, "the PC's memory is not known"),
    }
}

/// How the Series S profile's upscale is served.
#[derive(Debug, PartialEq, Eq)]
pub enum Upscaler {
    /// Not the Series S profile, or the user set an upscaler already.
    AsSet,
    /// AMD's FidelityFX DLLs are in `VICTORY_FFX_DIR`.
    Amd,
    /// No DLLs: the bilinear stand-in, so that the title does not stop.
    Simple,
}

/// The upscaler for `console`, from `VICTORY_FFX_DIR` (`ffx_dir`) and
/// whether `VICTORY_FFX_SIMPLE` is set (`simple_set`).
pub fn upscaler(console: Console, ffx_dir: Option<&Path>, simple_set: bool) -> Upscaler {
    if !matches!(console, Console::SeriesS) || simple_set {
        return Upscaler::AsSet;
    }
    match ffx_dir {
        Some(dir) if dir.join(FFX_LOADER).is_file() => Upscaler::Amd,
        _ => Upscaler::Simple,
    }
}

/// The PC's physical memory in bytes.
#[cfg(windows)]
pub fn physical_memory() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut m = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a sized MEMORYSTATUSEX for the call to fill.
    (unsafe { GlobalMemoryStatusEx(&mut m) } != 0).then_some(m.ullTotalPhys)
}

#[cfg(not(windows))]
pub fn physical_memory() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1 << 30;

    #[test]
    fn the_user_choice_stands() {
        for chosen in [Console::SeriesX, Console::SeriesS] {
            for physical in [None, Some(8 * GB), Some(64 * GB)] {
                let picked = console(Some(chosen), physical).0;
                assert_eq!(picked as u8, chosen as u8);
            }
        }
    }

    #[test]
    fn series_s_below_24_gb_series_x_from_it() {
        let pick = |gb: u64| console(None, Some(gb * GB)).0;
        // 16 GB machines report a little less than 16 GiB.
        assert!(matches!(pick(15), Console::SeriesS));
        assert!(matches!(pick(16), Console::SeriesS));
        assert!(matches!(pick(23), Console::SeriesS));
        assert!(matches!(pick(24), Console::SeriesX));
        assert!(matches!(pick(32), Console::SeriesX));
        assert!(matches!(console(None, None).0, Console::SeriesX));
    }

    #[test]
    fn series_s_gets_an_upscaler_that_is_there() {
        let dir = std::env::temp_dir().join(format!("victory-ffx-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // No DLL in the folder, or no folder named: the stand-in.
        assert_eq!(
            upscaler(Console::SeriesS, Some(&dir), false),
            Upscaler::Simple
        );
        assert_eq!(upscaler(Console::SeriesS, None, false), Upscaler::Simple);
        std::fs::write(dir.join(FFX_LOADER), b"").unwrap();
        assert_eq!(upscaler(Console::SeriesS, Some(&dir), false), Upscaler::Amd);
        // The user's VICTORY_FFX_SIMPLE, and Series X, are left alone.
        assert_eq!(
            upscaler(Console::SeriesS, Some(&dir), true),
            Upscaler::AsSet
        );
        assert_eq!(upscaler(Console::SeriesX, None, false), Upscaler::AsSet);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
