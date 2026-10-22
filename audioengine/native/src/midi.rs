//! A MIDI controller, read here so its messages never pass through the GUI's Python: midir's input thread
//! queues each message with the time it came (the shared clock) and wakes the engine thread, which acts on it
//! at once (deck.rs). The output port of the same name lights the controller's lamps (PLAY, CUE).

use std::sync::{Arc, Mutex};
use std::thread::Thread;

use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};

use crate::clock;

/// (time s on the shared clock, status, data1, data2: a note off as a note on of velocity 0)
pub type Msg = (f64, [u8; 3]);

pub struct Midi {
    _input: MidiInputConnection<()>,
    output: Option<MidiOutputConnection>,
    queue: Arc<Mutex<Vec<Msg>>>,
    pub name: String,
}

/// The input ports' names.
pub fn ports() -> Vec<String> {
    let Ok(input) = MidiInput::new("Mixlyzer") else { return vec![] };
    input.ports().iter().filter_map(|p| input.port_name(p).ok()).collect()
}

/// A channel message as the deck reads it: a note off is a note on of velocity 0; others (system messages)
/// None.
fn channel_message(bytes: &[u8]) -> Option<[u8; 3]> {
    let status = *bytes.first()?;
    if !(0x80..0xF0).contains(&status) {
        return None;
    }
    let d1 = bytes.get(1).copied().unwrap_or(0);
    let d2 = bytes.get(2).copied().unwrap_or(0);
    if status & 0xF0 == 0x80 {
        return Some([0x90 | (status & 0x0F), d1, 0]);
    }
    Some([status, d1, d2])
}

impl Midi {
    /// The input port ``name`` (else the first whose name holds it) and the output port of its name, if any.
    pub fn open(name: &str, wake: Option<Thread>) -> Result<Midi, String> {
        let mut input = MidiInput::new("Mixlyzer").map_err(|e| e.to_string())?;
        input.ignore(Ignore::All);
        let ports = input.ports();
        let names: Vec<String> = ports.iter().map(|p| input.port_name(p).unwrap_or_default()).collect();
        let at = names.iter().position(|n| n == name).or_else(|| names.iter().position(|n| n.contains(name)));
        let Some(at) = at else { return Err(format!("no MIDI input {name}")) };
        let port_name = names[at].clone();
        let queue = Arc::new(Mutex::new(Vec::new()));
        let q = queue.clone();
        let connection = input
            .connect(
                &ports[at],
                "Mixlyzer deck",
                move |_stamp, bytes, _| {
                    if let Some(msg) = channel_message(bytes) {
                        q.lock().unwrap().push((clock::now(), msg));
                        if let Some(t) = &wake {
                            t.unpark();
                        }
                    }
                },
                (),
            )
            .map_err(|e| e.to_string())?;
        let output = MidiOutput::new("Mixlyzer").ok().and_then(|out| {
            let ports = out.ports();
            let port = ports.iter().find(|p| out.port_name(p).map(|n| n == port_name).unwrap_or(false))?.clone();
            out.connect(&port, "Mixlyzer deck lamps").ok()
        });
        Ok(Midi { _input: connection, output, queue, name: port_name })
    }

    pub fn drain(&self) -> Vec<Msg> {
        std::mem::take(&mut *self.queue.lock().unwrap())
    }

    pub fn send(&mut self, msg: &[u8]) {
        if let Some(out) = self.output.as_mut() {
            let _ = out.send(msg);
        }
    }
}
