//! World-frame-indexed input recording and deterministic replay.
//!
//! Activated only by environment variables:
//! - `VICTORY_INPUT_RECORD=<path.jsonl>`: records `PadState` and `KeyState`
//!   changes tagged with `victory_gpu::bench::world_frame()`.
//! - `VICTORY_INPUT_REPLAY=<path.jsonl>`: replays recorded traces or
//!   declarative range lines (`{"from":0,"to":120,"ly":1.0,"rt":0.8}`) indexed
//!   by `world_frame()`, holding neutral while `world_frame() == 0` so loading
//!   screens never desynchronize replay.
//!
//! ## Delivery contract
//! - While `world_frame == 0` (loading screens / menus), replay holds neutral
//!   input and neither advances playback state nor consumes pulses.
//! - Recorded state-change traces (`wframe`) behave as step-function holds:
//!   the latest entry with `entry.wframe <= world_frame` remains active until
//!   the next recorded change.
//! - Stateless point lookups (`Script::pad_at`, `Script::keys_at`) evaluate
//!   declarative ranges strictly at `from <= world_frame <= to`.
//! - Stateful live playback (`ReplayPlayback` / `ScriptedSource`) guarantees
//!   once-only delivery for authored 1-frame pulses (`from == to`, `from >= 1`):
//!   if no poll occurs on world frame `from`, any unpolled pulse in the gap
//!   `last_polled_wframe < from <= world_frame` latches onto the first eligible
//!   polled world frame, stays consistent across all polls and pad/key reads
//!   during that single world frame, and transitions to consumed on the next
//!   world frame so it never repeats.
//! - Multi-frame declarative spans (`to > from`) never latch past `to`.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use super::gameinput::{KeyState, PadState, Source};

const PADS: usize = 4;

#[derive(Clone, Debug, PartialEq)]
struct TracePadEntry {
    wframe: u64,
    slot: usize,
    connected: bool,
    pad: PadState,
}

#[derive(Clone, Debug, PartialEq)]
struct TraceKeysEntry {
    wframe: u64,
    keys: Vec<KeyState>,
}

#[derive(Clone, Debug, Default, PartialEq)]
struct RangeEntry {
    from: u64,
    to: u64,
    slot: usize,
    buttons: Option<u32>,
    lt: Option<f32>,
    rt: Option<f32>,
    lx: Option<f32>,
    ly: Option<f32>,
    rx: Option<f32>,
    ry: Option<f32>,
    keys: Vec<KeyState>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Script {
    pad_traces: [Vec<TracePadEntry>; PADS],
    key_traces: Vec<TraceKeysEntry>,
    ranges: Vec<RangeEntry>,
}

impl Script {
    pub(crate) fn parse(text: &str) -> Result<Self, String> {
        let mut script = Self::default();
        for (line_no, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            let obj = parse_json_object(line)
                .map_err(|e| format!("line {}: {e}", line_no + 1))?;
            if obj.contains_key("from") || obj.contains_key("to") {
                let from = obj_u64(&obj, "from")?.unwrap_or(0);
                let to = obj_u64(&obj, "to")?.unwrap_or(u64::MAX);
                if to < from {
                    return Err(format!("line {}: `to` ({to}) < `from` ({from})", line_no + 1));
                }
                let slot = obj_u64(&obj, "slot")?.unwrap_or(0) as usize;
                if slot >= PADS {
                    return Err(format!("line {}: pad slot {slot} out of range", line_no + 1));
                }
                let keys = obj_keys(&obj, "keys")?;
                script.ranges.push(RangeEntry {
                    from,
                    to,
                    slot,
                    buttons: obj_u64(&obj, "buttons")?.map(|v| v as u32),
                    lt: obj_f32(&obj, "lt")?,
                    rt: obj_f32(&obj, "rt")?,
                    lx: obj_f32(&obj, "lx")?,
                    ly: obj_f32(&obj, "ly")?,
                    rx: obj_f32(&obj, "rx")?,
                    ry: obj_f32(&obj, "ry")?,
                    keys,
                });
                continue;
            }

            let wframe = obj_u64(&obj, "wframe")?
                .ok_or_else(|| format!("line {}: missing `wframe` or `from`/`to`", line_no + 1))?;
            if let Some(keys_val) = obj.get("keys")
                && !obj.contains_key("slot")
                && !obj.contains_key("connected")
                && !obj.contains_key("buttons")
            {
                let keys = parse_keys_value(keys_val)
                    .map_err(|e| format!("line {}: {e}", line_no + 1))?;
                script.key_traces.push(TraceKeysEntry { wframe, keys });
                continue;
            }

            let slot = obj_u64(&obj, "slot")?.unwrap_or(0) as usize;
            if slot >= PADS {
                return Err(format!("line {}: pad slot {slot} out of range", line_no + 1));
            }
            let connected = obj_bool(&obj, "connected")?.unwrap_or(true);
            let pad = PadState {
                buttons: obj_u64(&obj, "buttons")?.unwrap_or(0) as u32,
                left_trigger: obj_f32(&obj, "lt")?.unwrap_or(0.0),
                right_trigger: obj_f32(&obj, "rt")?.unwrap_or(0.0),
                left_x: obj_f32(&obj, "lx")?.unwrap_or(0.0),
                left_y: obj_f32(&obj, "ly")?.unwrap_or(0.0),
                right_x: obj_f32(&obj, "rx")?.unwrap_or(0.0),
                right_y: obj_f32(&obj, "ry")?.unwrap_or(0.0),
            };
            script.pad_traces[slot].push(TracePadEntry {
                wframe,
                slot,
                connected,
                pad,
            });
            if obj.contains_key("keys") {
                let keys = obj_keys(&obj, "keys")?;
                script.key_traces.push(TraceKeysEntry { wframe, keys });
            }
        }
        for entries in &mut script.pad_traces {
            entries.sort_by_key(|e| e.wframe);
        }
        script.key_traces.sort_by_key(|e| e.wframe);
        Ok(script)
    }

    fn pad_at_matching(
        &self,
        wframe: u64,
        slot: usize,
        mut range_active: impl FnMut(usize, &RangeEntry) -> bool,
    ) -> Option<PadState> {
        if slot >= PADS {
            return None;
        }
        // Hold neutral during loading screens (`wframe == 0`), keeping slot 0
        // connected so the title does not show a disconnect prompt.
        if wframe == 0 {
            return (slot == 0).then(PadState::default);
        }
        let mut state = match self.pad_traces[slot].partition_point(|e| e.wframe <= wframe) {
            0 => (slot == 0).then(PadState::default),
            idx => {
                let entry = &self.pad_traces[slot][idx - 1];
                entry.connected.then_some(entry.pad)
            }
        };
        for (idx, range) in self.ranges.iter().enumerate() {
            if range.slot == slot && range_active(idx, range) {
                let pad = state.get_or_insert_with(PadState::default);
                if let Some(b) = range.buttons {
                    pad.buttons |= b;
                }
                if let Some(v) = range.lt {
                    pad.left_trigger = v;
                }
                if let Some(v) = range.rt {
                    pad.right_trigger = v;
                }
                if let Some(v) = range.lx {
                    pad.left_x = v;
                }
                if let Some(v) = range.ly {
                    pad.left_y = v;
                }
                if let Some(v) = range.rx {
                    pad.right_x = v;
                }
                if let Some(v) = range.ry {
                    pad.right_y = v;
                }
            }
        }
        state
    }

    fn keys_at_matching(
        &self,
        wframe: u64,
        mut range_active: impl FnMut(usize, &RangeEntry) -> bool,
    ) -> Vec<KeyState> {
        if wframe == 0 {
            return Vec::new();
        }
        let mut keys = match self.key_traces.partition_point(|e| e.wframe <= wframe) {
            0 => Vec::new(),
            idx => self.key_traces[idx - 1].keys.clone(),
        };
        for (idx, range) in self.ranges.iter().enumerate() {
            if range_active(idx, range) {
                for &k in &range.keys {
                    if !keys.contains(&k) {
                        keys.push(k);
                    }
                }
            }
        }
        keys
    }

    /// Stateless point-in-time gamepad lookup at exact `wframe` (`from <= wframe <= to`).
    ///
    /// For stateful replay with zero-poll 1-frame pulse latching (`from == to`),
    /// use [`ReplayPlayback`] or [`ScriptedSource`].
    #[cfg(test)]
    pub(crate) fn pad_at(&self, wframe: u64, slot: usize) -> Option<PadState> {
        self.pad_at_matching(wframe, slot, |_, r| r.from <= wframe && wframe <= r.to)
    }

    /// Stateless point-in-time keyboard lookup at exact `wframe` (`from <= wframe <= to`).
    ///
    /// For stateful replay with zero-poll 1-frame pulse latching (`from == to`),
    /// use [`ReplayPlayback`] or [`ScriptedSource`].
    #[cfg(test)]
    pub(crate) fn keys_at(&self, wframe: u64) -> Vec<KeyState> {
        self.keys_at_matching(wframe, |_, r| r.from <= wframe && wframe <= r.to)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PulseState {
    /// Authored 1-frame pulse (`from == to`) has not yet been delivered.
    Pending,
    /// Latched and active for all polls on this single world frame.
    ActiveOn(u64),
    /// Already delivered on an earlier world frame; must never repeat.
    Consumed,
}

fn range_is_active(
    pulse_states: &[PulseState],
    idx: usize,
    range: &RangeEntry,
    wframe: u64,
) -> bool {
    if range.from == range.to {
        pulse_states[idx] == PulseState::ActiveOn(wframe)
    } else {
        range.from <= wframe && wframe <= range.to
    }
}

/// Stateful world-frame replay cursor over a [`Script`] that latches unpolled
/// 1-frame declarative pulses (`from == to`, `from >= 1`) across zero-poll gaps
/// so each pulse delivers once on the first eligible world frame and never repeats.
#[derive(Clone, Debug)]
pub(crate) struct ReplayPlayback {
    script: Script,
    last_polled_wframe: u64,
    pulse_states: Vec<PulseState>,
}

impl ReplayPlayback {
    pub(crate) fn new(script: Script) -> Self {
        let pulse_states = vec![PulseState::Pending; script.ranges.len()];
        Self {
            script,
            last_polled_wframe: 0,
            pulse_states,
        }
    }

    fn advance_to(&mut self, wframe: u64) {
        if wframe == 0 || wframe <= self.last_polled_wframe {
            return;
        }
        let prev = self.last_polled_wframe;
        for (state, range) in self.pulse_states.iter_mut().zip(&self.script.ranges) {
            if range.from != range.to {
                continue;
            }
            match *state {
                PulseState::ActiveOn(delivered_wf) if delivered_wf < wframe => {
                    *state = PulseState::Consumed;
                }
                PulseState::Pending if prev < range.from && range.from <= wframe => {
                    *state = PulseState::ActiveOn(wframe);
                }
                _ => {}
            }
        }
        self.last_polled_wframe = wframe;
    }

    pub(crate) fn pad_at(&mut self, wframe: u64, slot: usize) -> Option<PadState> {
        self.advance_to(wframe);
        let states = &self.pulse_states;
        self.script
            .pad_at_matching(wframe, slot, |idx, r| range_is_active(states, idx, r, wframe))
    }

    pub(crate) fn keys_at(&mut self, wframe: u64) -> Vec<KeyState> {
        self.advance_to(wframe);
        let states = &self.pulse_states;
        self.script
            .keys_at_matching(wframe, |idx, r| range_is_active(states, idx, r, wframe))
    }
}

struct Recorder {
    writer: BufWriter<File>,
    started: Instant,
    last_pads: [Option<Option<PadState>>; PADS],
    last_keys: Option<Vec<KeyState>>,
}

impl Recorder {
    fn create(path: &Path) -> Result<Self, String> {
        victory_gpu::bench::ensure_outside_repo(path)?;
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("cannot create parent dir for {}: {e}", path.display()))?;
        }
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(path)
            .map_err(|e| format!("cannot open {} for input recording: {e}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
            started: Instant::now(),
            last_pads: [None; PADS],
            last_keys: None,
        })
    }

    fn record_pad(&mut self, wframe: u64, slot: usize, pad: Option<PadState>) {
        if wframe == 0 || slot >= PADS {
            return;
        }
        if self.last_pads[slot].as_ref() == Some(&pad) {
            return;
        }
        self.last_pads[slot] = Some(pad);
        let ms = self.started.elapsed().as_millis() as u64;
        let line = match pad {
            Some(p) => format!(
                "{{\"wframe\":{},\"ms\":{},\"slot\":{},\"connected\":true,\"buttons\":{},\"lt\":{},\"rt\":{},\"lx\":{},\"ly\":{},\"rx\":{},\"ry\":{}}}\n",
                wframe,
                ms,
                slot,
                p.buttons,
                format_f32(p.left_trigger),
                format_f32(p.right_trigger),
                format_f32(p.left_x),
                format_f32(p.left_y),
                format_f32(p.right_x),
                format_f32(p.right_y),
            ),
            None => format!(
                "{{\"wframe\":{wframe},\"ms\":{ms},\"slot\":{slot},\"connected\":false}}\n"
            ),
        };
        let _ = self.writer.write_all(line.as_bytes());
        let _ = self.writer.flush();
    }

    fn record_keys(&mut self, wframe: u64, keys: &[KeyState]) {
        if wframe == 0 {
            return;
        }
        if self.last_keys.as_deref() == Some(keys) {
            return;
        }
        self.last_keys = Some(keys.to_vec());
        let ms = self.started.elapsed().as_millis() as u64;
        let mut line = format!("{{\"wframe\":{wframe},\"ms\":{ms},\"keys\":[");
        for (i, k) in keys.iter().enumerate() {
            if i > 0 {
                line.push(',');
            }
            use std::fmt::Write;
            let _ = write!(
                line,
                "{{\"vk\":{},\"scan\":{},\"cp\":{},\"dead\":{}}}",
                k.virtual_key, k.scan_code, k.code_point, k.dead_key
            );
        }
        line.push_str("]}\n");
        let _ = self.writer.write_all(line.as_bytes());
        let _ = self.writer.flush();
    }
}

fn format_f32(v: f32) -> String {
    if !v.is_finite() {
        return "0.0".to_owned();
    }
    let s = v.to_string();
    if s.contains('.') || s.contains('e') || s.contains('E') {
        s
    } else {
        format!("{s}.0")
    }
}

pub(crate) struct ScriptedSource {
    inner: Box<dyn Source>,
    replay: Option<ReplayPlayback>,
    recorder: Option<Recorder>,
    world_frame: Box<dyn Fn() -> u64 + Send>,
    /// World frame latched on `pad(0)` during replay so all pad slots and
    /// `keys()` in the same `Hub::poll` cycle observe one consistent frame.
    latched_replay_wframe: Option<u64>,
}

impl ScriptedSource {
    fn with_fn(
        inner: Box<dyn Source>,
        replay: Option<Script>,
        recorder: Option<Recorder>,
        world_frame: impl Fn() -> u64 + Send + 'static,
    ) -> Self {
        Self {
            inner,
            replay: replay.map(ReplayPlayback::new),
            recorder,
            world_frame: Box::new(world_frame),
            latched_replay_wframe: None,
        }
    }
}

impl Source for ScriptedSource {
    fn pad(&mut self, slot: usize) -> Option<PadState> {
        let (wf, pad) = match &mut self.replay {
            Some(playback) => {
                let wf = if slot == 0 {
                    let wf = (self.world_frame)();
                    self.latched_replay_wframe = Some(wf);
                    wf
                } else {
                    self.latched_replay_wframe
                        .unwrap_or_else(|| (self.world_frame)())
                };
                (wf, playback.pad_at(wf, slot))
            }
            None => ((self.world_frame)(), self.inner.pad(slot)),
        };
        if let Some(rec) = &mut self.recorder {
            rec.record_pad(wf, slot, pad);
        }
        pad
    }

    fn keys(&mut self) -> Vec<KeyState> {
        let (wf, keys) = match &mut self.replay {
            Some(playback) => {
                let wf = self
                    .latched_replay_wframe
                    .take()
                    .unwrap_or_else(|| (self.world_frame)());
                (wf, playback.keys_at(wf))
            }
            None => ((self.world_frame)(), self.inner.keys()),
        };
        if let Some(rec) = &mut self.recorder {
            rec.record_keys(wf, &keys);
        }
        keys
    }

    fn rumble(&mut self, slot: usize, low: u16, high: u16) {
        if self.replay.is_none() {
            self.inner.rumble(slot, low, high);
        }
    }
}

/// Wraps `source` with input replay and/or recording when `VICTORY_INPUT_REPLAY`
/// or `VICTORY_INPUT_RECORD` is configured.
pub(crate) fn wrap(source: Box<dyn Source>) -> Box<dyn Source> {
    let replay_path = std::env::var_os("VICTORY_INPUT_REPLAY")
        .map(|v| v.to_string_lossy().trim().to_owned())
        .filter(|s| !s.is_empty());
    let record_path = std::env::var_os("VICTORY_INPUT_RECORD")
        .map(|v| v.to_string_lossy().trim().to_owned())
        .filter(|s| !s.is_empty());

    if replay_path.is_none() && record_path.is_none() {
        return source;
    }

    let replay = replay_path.and_then(|path| match std::fs::read(&path) {
        Ok(bytes) => {
            let sha = victory_gpu::bench::register_input_script(&bytes);
            match std::str::from_utf8(&bytes)
                .map_err(|e| format!("invalid UTF-8: {e}"))
                .and_then(Script::parse)
            {
                Ok(script) => {
                    tracing::info!(path, sha256 = %sha, "replaying scripted world-frame inputs");
                    Some(script)
                }
                Err(err) => {
                    tracing::error!(path, error = %err, "failed to parse VICTORY_INPUT_REPLAY");
                    None
                }
            }
        }
        Err(err) => {
            tracing::error!(path, error = %err, "failed to read VICTORY_INPUT_REPLAY");
            None
        }
    });

    let recorder = record_path.and_then(|path| match Recorder::create(Path::new(&path)) {
        Ok(rec) => {
            tracing::info!(path, "recording world-frame inputs");
            Some(rec)
        }
        Err(err) => {
            tracing::error!(path, error = %err, "failed to open VICTORY_INPUT_RECORD");
            None
        }
    });

    if replay.is_none() && recorder.is_none() {
        return source;
    }

    Box::new(ScriptedSource::with_fn(
        source,
        replay,
        recorder,
        victory_gpu::bench::world_frame,
    ))
}

#[derive(Clone, Debug)]
enum JsonVal {
    Bool(bool),
    Num(String),
    Array(Vec<JsonVal>),
    Object(HashMap<String, JsonVal>),
    Null,
    Str(String),
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            bytes: input.as_bytes(),
            pos: 0,
        }
    }

    fn skip_ws(&mut self) {
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_whitespace() {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, ch: u8) -> Result<(), String> {
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&ch) {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("expected '{}' at byte {}", ch as char, self.pos))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect(b'"')?;
        let mut out = Vec::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            match b {
                b'"' => {
                    return String::from_utf8(out)
                        .map_err(|e| format!("invalid UTF-8 in JSON string: {e}"));
                }
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| "unterminated escape".to_owned())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push(b'"'),
                        b'\\' => out.push(b'\\'),
                        b'/' => out.push(b'/'),
                        b'n' => out.push(b'\n'),
                        b'r' => out.push(b'\r'),
                        b't' => out.push(b'\t'),
                        _ => return Err(format!("unsupported escape \\{}", esc as char)),
                    }
                }
                c => out.push(c),
            }
        }
        Err("unterminated string".to_owned())
    }

    fn parse_number(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(&b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(format!("expected number at byte {}", self.pos));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .map(str::to_owned)
            .map_err(|e| e.to_string())
    }

    fn parse_value(&mut self) -> Result<JsonVal, String> {
        self.skip_ws();
        let Some(&b) = self.bytes.get(self.pos) else {
            return Err("unexpected end of JSON".to_owned());
        };
        match b {
            b'{' => Ok(JsonVal::Object(self.parse_object()?)),
            b'[' => {
                self.pos += 1;
                let mut items = Vec::new();
                self.skip_ws();
                if self.bytes.get(self.pos) == Some(&b']') {
                    self.pos += 1;
                    return Ok(JsonVal::Array(items));
                }
                loop {
                    items.push(self.parse_value()?);
                    self.skip_ws();
                    match self.bytes.get(self.pos) {
                        Some(&b',') => self.pos += 1,
                        Some(&b']') => {
                            self.pos += 1;
                            return Ok(JsonVal::Array(items));
                        }
                        _ => return Err(format!("expected ',' or ']' at byte {}", self.pos)),
                    }
                }
            }
            b'"' => Ok(JsonVal::Str(self.parse_string()?)),
            b't' if self.bytes[self.pos..].starts_with(b"true") => {
                self.pos += 4;
                Ok(JsonVal::Bool(true))
            }
            b'f' if self.bytes[self.pos..].starts_with(b"false") => {
                self.pos += 5;
                Ok(JsonVal::Bool(false))
            }
            b'n' if self.bytes[self.pos..].starts_with(b"null") => {
                self.pos += 4;
                Ok(JsonVal::Null)
            }
            _ => Ok(JsonVal::Num(self.parse_number()?)),
        }
    }

    fn parse_object(&mut self) -> Result<HashMap<String, JsonVal>, String> {
        self.expect(b'{')?;
        let mut map = HashMap::new();
        self.skip_ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(map);
        }
        loop {
            let key = self.parse_string()?;
            self.expect(b':')?;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            match self.bytes.get(self.pos) {
                Some(&b',') => self.pos += 1,
                Some(&b'}') => {
                    self.pos += 1;
                    return Ok(map);
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.pos)),
            }
        }
    }
}

fn parse_json_object(line: &str) -> Result<HashMap<String, JsonVal>, String> {
    let mut parser = JsonParser::new(line);
    let obj = parser.parse_object()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(format!("trailing characters at byte {}", parser.pos));
    }
    Ok(obj)
}

fn obj_u64(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<u64>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Num(s)) => s
            .parse::<u64>()
            .map(Some)
            .map_err(|e| format!("invalid integer `{key}` ({s}): {e}")),
        _ => Err(format!("expected integer for `{key}`")),
    }
}

fn obj_f32(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<f32>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Num(s)) => s
            .parse::<f32>()
            .map(Some)
            .map_err(|e| format!("invalid float `{key}` ({s}): {e}")),
        _ => Err(format!("expected float for `{key}`")),
    }
}

fn obj_bool(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Option<bool>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(None),
        Some(JsonVal::Bool(b)) => Ok(Some(*b)),
        _ => Err(format!("expected bool for `{key}`")),
    }
}

fn obj_keys(obj: &HashMap<String, JsonVal>, key: &str) -> Result<Vec<KeyState>, String> {
    match obj.get(key) {
        None | Some(JsonVal::Null) => Ok(Vec::new()),
        Some(v) => parse_keys_value(v),
    }
}

fn parse_keys_value(val: &JsonVal) -> Result<Vec<KeyState>, String> {
    let JsonVal::Array(arr) = val else {
        return Err("expected array for `keys`".to_owned());
    };
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        match item {
            JsonVal::Object(map) => {
                let vk = obj_u64(map, "vk")?
                    .or(obj_u64(map, "virtual_key")?)
                    .ok_or_else(|| "key object missing `vk`".to_owned())? as u8;
                let scan = obj_u64(map, "scan")?
                    .or(obj_u64(map, "scan_code")?)
                    .unwrap_or(0) as u32;
                let cp = obj_u64(map, "cp")?
                    .or(obj_u64(map, "code_point")?)
                    .unwrap_or(0) as u32;
                let dead = obj_bool(map, "dead")?
                    .or(obj_bool(map, "dead_key")?)
                    .unwrap_or(false);
                out.push(KeyState {
                    scan_code: scan,
                    code_point: cp,
                    virtual_key: vk,
                    dead_key: dead,
                });
            }
            JsonVal::Num(s) => {
                let vk = s
                    .parse::<u8>()
                    .map_err(|e| format!("invalid key code {s}: {e}"))?;
                out.push(KeyState {
                    virtual_key: vk,
                    ..Default::default()
                });
            }
            JsonVal::Str(s) if s.len() == 1 => {
                out.push(KeyState {
                    virtual_key: s.as_bytes()[0].to_ascii_uppercase(),
                    ..Default::default()
                });
            }
            _ => return Err("unsupported entry inside `keys`".to_owned()),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    use super::*;

    #[derive(Default)]
    struct MockHost {
        pads: [Option<PadState>; PADS],
        keys: Vec<KeyState>,
    }

    struct MockSource(Arc<Mutex<MockHost>>);

    impl Source for MockSource {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            self.0.lock().unwrap().pads[slot]
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.0.lock().unwrap().keys.clone()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }

    static TEST_WFRAME: AtomicU64 = AtomicU64::new(0);
    fn test_world_frame() -> u64 {
        TEST_WFRAME.load(Ordering::Relaxed)
    }

    #[test]
    fn record_then_replay_roundtrips_pad_state_and_keys_bit_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("trace.jsonl");
        let host = Arc::new(Mutex::new(MockHost::default()));

        let rec = Recorder::create(&path).unwrap();
        let mut src = ScriptedSource::with_fn(
            Box::new(MockSource(Arc::clone(&host))),
            None,
            Some(rec),
            test_world_frame,
        );

        let pad_a = PadState {
            buttons: 0x0405,
            left_trigger: 0.125,
            right_trigger: 0.8125,
            left_x: -0.33333334,
            left_y: 0.9999695,
            right_x: 0.25,
            right_y: -0.75,
        };
        let pad_b = PadState {
            buttons: 0x20,
            left_trigger: 0.0,
            right_trigger: 1.0,
            left_x: 0.0,
            left_y: 1.0,
            right_x: -1.0,
            right_y: 0.0,
        };
        let key_w = KeyState {
            scan_code: 0x11,
            code_point: 0,
            virtual_key: b'W',
            dead_key: false,
        };

        // Loading screen (`wframe == 0`) must not record any trace events.
        TEST_WFRAME.store(0, Ordering::Relaxed);
        host.lock().unwrap().pads[0] = Some(pad_a);
        assert_eq!(src.pad(0), Some(pad_a));
        assert!(std::fs::read_to_string(&path).unwrap().is_empty());

        // Frame 1 & Frame 10 record changes.
        TEST_WFRAME.store(1, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(pad_a));
        host.lock().unwrap().keys = vec![key_w];
        assert_eq!(src.keys(), vec![key_w]);

        TEST_WFRAME.store(10, Ordering::Relaxed);
        host.lock().unwrap().pads[0] = Some(pad_b);
        host.lock().unwrap().keys.clear();
        assert_eq!(src.pad(0), Some(pad_b));
        assert!(src.keys().is_empty());
        drop(src);

        let raw = std::fs::read_to_string(&path).unwrap();
        let script = Script::parse(&raw).unwrap();

        // Loading screen holds neutral pad 0 and empty keys.
        assert_eq!(script.pad_at(0, 0), Some(PadState::default()));
        assert!(script.keys_at(0).is_empty());

        let got_1 = script.pad_at(1, 0).unwrap();
        assert_eq!(got_1.buttons, pad_a.buttons);
        assert_eq!(got_1.left_x.to_bits(), pad_a.left_x.to_bits());
        assert_eq!(got_1.left_y.to_bits(), pad_a.left_y.to_bits());
        assert_eq!(got_1.right_trigger.to_bits(), pad_a.right_trigger.to_bits());
        assert_eq!(script.keys_at(5), vec![key_w]);

        let got_10 = script.pad_at(10, 0).unwrap();
        assert_eq!(got_10.buttons, pad_b.buttons);
        assert_eq!(got_10.left_y.to_bits(), pad_b.left_y.to_bits());
        assert!(script.keys_at(10).is_empty());
    }

    #[test]
    fn declarative_range_scripts_wait_on_loading_screen_and_apply_in_world() {
        let text = concat!(
            "# drive forward for 120 world frames\n",
            "{\"from\":0,\"to\":120,\"ly\":1.0,\"rt\":0.8}\n",
            "{\"from\":50,\"to\":60,\"buttons\":4,\"keys\":[87]}\n",
        );
        let script = Script::parse(text).unwrap();

        // Loading screen (`wframe == 0`) stays neutral.
        assert_eq!(script.pad_at(0, 0), Some(PadState::default()));
        assert!(script.keys_at(0).is_empty());

        let p1 = script.pad_at(1, 0).unwrap();
        assert_eq!(p1.left_y, 1.0);
        assert_eq!(p1.right_trigger, 0.8);
        assert_eq!(p1.buttons, 0);

        let p55 = script.pad_at(55, 0).unwrap();
        assert_eq!(p55.left_y, 1.0);
        assert_eq!(p55.right_trigger, 0.8);
        assert_eq!(p55.buttons, 4);
        assert_eq!(script.keys_at(55).len(), 1);
        assert_eq!(script.keys_at(55)[0].virtual_key, 87);

        // After `to`, returns to rest.
        assert_eq!(script.pad_at(121, 0), Some(PadState::default()));
        assert!(script.keys_at(121).is_empty());

        // Recording rejects paths inside any `.victory-root` repository tree.
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("input-trace.jsonl");
        assert!(Recorder::create(&inside).is_err());

        // Multi-byte UTF-8 characters inside JSON strings decode without Latin-1 corruption.
        let utf8_script = Script::parse("{\"from\":1,\"to\":2,\"note\":\"тест — ✓\",\"ly\":0.5}\n").unwrap();
        assert_eq!(utf8_script.pad_at(1, 0).unwrap().left_y, 0.5);
    }

    #[test]
    fn variable_fps_replay_produces_identical_inputs_per_world_frame() {
        let script = Script::parse(concat!(
            "{\"from\":1,\"to\":5,\"ly\":1.0}\n",
            "{\"from\":6,\"to\":10,\"rx\":0.5,\"rt\":0.75}\n",
        ))
        .unwrap();

        let simulate = |loading_polls: usize, polls_per_world_frame: usize| {
            let wf = Arc::new(AtomicU64::new(0));
            let mut out = Vec::new();
            for _ in 0..loading_polls {
                out.push((0, script.pad_at(wf.load(Ordering::Relaxed), 0)));
            }
            for frame in 1..=11u64 {
                wf.store(frame, Ordering::Relaxed);
                for _ in 0..polls_per_world_frame {
                    out.push((frame, script.pad_at(frame, 0)));
                }
            }
            out
        };

        let slow_20fps = simulate(15, 1);
        let fast_120fps = simulate(3, 4);
        for frame in 1..=11u64 {
            let a: Vec<_> = slow_20fps
                .iter()
                .filter(|(f, _)| *f == frame)
                .map(|(_, p)| *p)
                .collect();
            let b: Vec<_> = fast_120fps
                .iter()
                .filter(|(f, _)| *f == frame)
                .map(|(_, p)| *p)
                .collect();
            assert!(!a.is_empty() && !b.is_empty());
            assert!(a.iter().all(|p| *p == a[0]));
            assert!(b.iter().all(|p| *p == a[0]));
        }
    }

    fn replay_source_with_clock(script: Script, clock: Arc<AtomicU64>) -> ScriptedSource {
        ScriptedSource::with_fn(
            Box::new(MockSource(Arc::new(Mutex::new(MockHost::default())))),
            Some(script),
            None,
            move || clock.load(Ordering::Relaxed),
        )
    }

    #[test]
    fn one_frame_pulse_latches_across_zero_poll_gap_and_delivers_once() {
        let script = Script::parse("{\"from\":2,\"to\":2,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut src = replay_source_with_clock(script, Arc::clone(&clock));

        // Poll frames [1, 3, 4]: frame 2 has zero polls, so the 1-frame pulse
        // latches onto frame 3, stays consistent within frame 3, and clears on frame 4.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());

        clock.store(3, Ordering::Relaxed);
        let p3_first = src.pad(0).unwrap();
        assert_eq!(p3_first.left_y, 1.0);
        let k3_first = src.keys();
        assert_eq!(k3_first.len(), 1);
        assert_eq!(k3_first[0].virtual_key, 87);

        // Repeated poll within the same delivery world frame keeps the latched pulse active.
        assert_eq!(src.pad(0).unwrap().left_y, 1.0);
        assert_eq!(src.keys().len(), 1);

        // Advancing past the delivery world frame releases the pulse permanently.
        clock.store(4, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());

        clock.store(5, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());

        // First world frame skipped after loading screen (`[0, 2, 3]` for `from=1, to=1`).
        let first_frame_pulse = Script::parse("{\"from\":1,\"to\":1,\"ly\":1.0}\n").unwrap();
        let clock2 = Arc::new(AtomicU64::new(0));
        let mut src2 = replay_source_with_clock(first_frame_pulse, Arc::clone(&clock2));
        assert_eq!(src2.pad(0), Some(PadState::default()));
        clock2.store(2, Ordering::Relaxed);
        assert_eq!(src2.pad(0).unwrap().left_y, 1.0);
        clock2.store(3, Ordering::Relaxed);
        assert_eq!(src2.pad(0), Some(PadState::default()));

        // Loading-screen pulse (`from=0, to=0`) never leaks into world frame 1.
        let zero_pulse = Script::parse("{\"from\":0,\"to\":0,\"ly\":1.0}\n").unwrap();
        let clock3 = Arc::new(AtomicU64::new(0));
        let mut src3 = replay_source_with_clock(zero_pulse, Arc::clone(&clock3));
        assert_eq!(src3.pad(0), Some(PadState::default()));
        clock3.store(1, Ordering::Relaxed);
        assert_eq!(src3.pad(0), Some(PadState::default()));
    }

    #[test]
    fn one_frame_pulse_normal_poll_delivers_once_on_target_frame() {
        let script = Script::parse("{\"from\":2,\"to\":2,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut src = replay_source_with_clock(script, Arc::clone(&clock));

        // Normal poll sequence [1, 2, 3]: delivers on frame 2, neutral on 1 & 3.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());

        clock.store(2, Ordering::Relaxed);
        assert_eq!(src.pad(0).unwrap().left_y, 1.0);
        assert_eq!(src.keys().len(), 1);
        assert_eq!(src.pad(0).unwrap().left_y, 1.0);
        assert_eq!(src.keys().len(), 1);

        clock.store(3, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());
    }

    #[test]
    fn multi_frame_span_does_not_latch_past_to_across_zero_poll_gap() {
        let script = Script::parse("{\"from\":2,\"to\":3,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut src = replay_source_with_clock(script, Arc::clone(&clock));

        // Multi-frame span (to > from) polled at [1, 4] must NOT latch past `to`.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());

        clock.store(4, Ordering::Relaxed);
        assert_eq!(src.pad(0), Some(PadState::default()));
        assert!(src.keys().is_empty());
    }
}
