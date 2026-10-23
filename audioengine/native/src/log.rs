//! The engine's console lines ("[AudioEngine] ..." on stderr), written by a thread of their own. The GUI reads them
//! through a pipe (into its log and the startup splash), and a pipe whose reader is busy blocks the writer: never the
//! engine thread or an output's callback, which only queue the line. Before `start` (the offline renderers) a line is
//! written at once.

use std::io::Write;
use std::sync::OnceLock;
use std::sync::mpsc::{Sender, channel};

static LINES: OnceLock<Sender<String>> = OnceLock::new();

/// The writer thread started: the lines from here on are queued to it.
pub fn start() {
    let (tx, rx) = channel::<String>();
    if LINES.set(tx).is_ok() {
        let _ = std::thread::Builder::new().name("log".into()).spawn(move || {
            let mut err = std::io::stderr();
            for line in rx {
                let _ = writeln!(err, "{line}");
                let _ = err.flush();
            }
        });
    }
}

pub fn line(text: String) {
    let text = format!("[AudioEngine] {text}");
    match LINES.get() {
        Some(tx) => {
            let _ = tx.send(text);
        }
        None => eprintln!("{text}"),
    }
}

/// A console line (format!'s arguments), "[AudioEngine] " before it.
macro_rules! log_line {
    ($($t:tt)*) => { $crate::log::line(format!($($t)*)) };
}
