//! Neuromorphic Learning Trainer
//!
//! This module provides the infrastructure for "Off-line Training" (Path B).
//! It replays recorded telemetry data from the Gold Dataset
//! into the Spiking Inference Engine to evolve thresholds, weights,
//! and connectivity via STDP.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::snn::SpikingInferenceEngine;
use crate::research::NeuromorphicSnapshot;

/// The Trainer manages the evolution of the SNN parameters.
///
/// ANALOGY:
/// If the `SpikingInferenceEngine` is the Brain, this Trainer is the "Experience".
/// It exposes the brain to past electrical events so it can learn patterns (Long-Term Potentiation).
pub struct NeuromorphicTrainer {
    pub engine: SpikingInferenceEngine,
}

impl NeuromorphicTrainer {
    pub fn new() -> Self {
        Self {
            engine: SpikingInferenceEngine::new(),
        }
    }

    /// Replays the research/neuromorphic_data.jsonl Gold Dataset.
    ///
    /// This is the primary way the AI "dreams" about hardware data
    /// to refine its internal models before being used for live control.
    /// Now evolves both thresholds AND synaptic weights via STDP.
    pub fn run_training_session<P: AsRef<Path>>(&mut self, data_path: P) -> Result<TrainingSummary, Box<dyn std::error::Error>> {
        let file = File::open(data_path)?;
        let reader = BufReader::new(file);

        let mut summary = TrainingSummary::default();
        let mut initial_thresholds = Vec::new();
        let mut initial_weights = Vec::new();

        for n in &self.engine.neurons {
            initial_thresholds.push(n.threshold);
            initial_weights.push(n.weights.clone());
            // One counter slot per neuron, initialized to zero
            summary.per_neuron_spikes.push(0);
        }

        for line in reader.lines() {
            let line = line?;
            if let Ok(entry) = serde_json::from_str::<NeuromorphicSnapshot>(&line) {
                // 1. Advance the neural state based on recorded electricity
                //    (STDP weight updates happen automatically inside step())
                let asset_deltas = [0.0; 7];
                self.engine.step(&asset_deltas, &entry.telemetry);

                // 2. Monitor Learning Progress
                summary.steps_processed += 1;

                // Track total and per-neuron activity
                for (idx, neuron) in self.engine.neurons.iter().enumerate() {
                    if neuron.last_spike {
                        summary.total_spikes += 1;
                        if let Some(count) = summary.per_neuron_spikes.get_mut(idx) {
                            *count += 1;
                        }
                    }
                }
            }
        }

        // Calculate deltas (Learning Progress)
        for (i, n) in self.engine.neurons.iter().enumerate() {
            // Threshold drift
            let delta = n.threshold - initial_thresholds[i];
            summary.threshold_drifts.push(delta);

            // Weight drift (per channel)
            let mut w_deltas = Vec::new();
            for (ch, &w) in n.weights.iter().enumerate() {
                let initial_w = initial_weights.get(i)
                    .and_then(|ws| ws.get(ch))
                    .copied()
                    .unwrap_or(1.0);
                w_deltas.push(w - initial_w);
            }
            summary.weight_drifts.push(w_deltas);
        }

        // PERSIST the learned brain to a file
        let _ = self.engine.save_parameters("DATA/research/student_model.json");

        Ok(summary)
    }

    /// EXPORT: Converts the learned thresholds AND weights into Verilog-compatible hex memory files.
    /// This allows the FPGA to "download" the learning from this Rust session.
    ///
    /// Format:
    ///   parameters.mem      — thresholds in Q8.8 fixed point
    ///   parameters_weights.mem — weight matrix in Q8.8 fixed point
    pub fn export_to_verilog<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        // Export thresholds
        let mut content = String::new();
        for neuron in &self.engine.neurons {
            // Convert 0.0-2.0 float range to 16-bit fixed point (Q8.8)
            // 1.0 = 256 (0x0100)
            let fixed_val = (neuron.threshold * 256.0) as u16;
            content.push_str(&format!("{:04X}\n", fixed_val));
        }
        std::fs::write(&path, content)?;

        // Export weight matrix (same Q8.8 format, row-major)
        let weight_path = path.as_ref().with_file_name("parameters_weights.mem");
        let mut w_content = String::new();
        for neuron in &self.engine.neurons {
            for &w in &neuron.weights {
                let fixed_w = (w * 256.0) as u16;
                w_content.push_str(&format!("{:04X}\n", fixed_w));
            }
        }
        std::fs::write(weight_path, w_content)?;

        Ok(())
    }

    /// SIMULATION: Exports stimulus and expected spikes in 16-bit packed hex format
    /// based on a replay of the provided data path.
    pub fn export_simulation_data<P: AsRef<Path>>(&mut self, data_path: P, stim_file: &str, exp_file: &str) -> std::io::Result<()> {
        let file = File::open(data_path)?;
        let reader = BufReader::new(file);

        let mut stim_content = String::new();
        let mut exp_content = String::new();

        for line in reader.lines() {
            let line = line.map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
            if let Ok(entry) = serde_json::from_str::<NeuromorphicSnapshot>(&line) {
                // 1. Pack Stimulus (Vcore, Power, Hashrate)
                // Format: [Vcore_8.8][Power_12.4][Hash_8.8]
                let vgfx = (entry.telemetry.vddcr_gfx_v * 256.0) as u16;
                let pwr = (entry.telemetry.power_w * 16.0) as u16;
                let hsh = (entry.telemetry.hashrate_mh * 256.0) as u16;

                stim_content.push_str(&format!("{:04X}{:04X}{:04X}\n", vgfx, pwr, hsh));

                // 2. Step engine and pack spikes
                let asset_deltas = [0.0; 7];
                self.engine.step(&asset_deltas, &entry.telemetry);
                let mut spike_byte: u8 = 0;
                for (i, neuron) in self.engine.neurons.iter().enumerate() {
                    if i < 8 && neuron.last_spike {
                        spike_byte |= 1 << i;
                    }
                }
                exp_content.push_str(&format!("{:02X}\n", spike_byte));
            }
        }

        std::fs::write(stim_file, stim_content)?;
        std::fs::write(exp_file, exp_content)?;
        Ok(())
    }
}

#[derive(Debug, Default)]
pub struct TrainingSummary {
    pub steps_processed: usize,
    pub total_spikes: u64,
    pub threshold_drifts: Vec<f32>,
    /// Per-neuron, per-channel weight deltas: weight_drifts[neuron][channel]
    pub weight_drifts: Vec<Vec<f32>>,
    /// Individual spike count per neuron across the full training session
    pub per_neuron_spikes: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_replay() {
        let mut trainer = NeuromorphicTrainer::new();
        let path = "DATA/research/neuromorphic_data.jsonl";

        // Only run if the file exists (prevents CI failure)
        if Path::new(path).exists() {
            let result = trainer.run_training_session(path);
            assert!(result.is_ok());
            let summary = result.unwrap();
            println!("Summary: {:?}", summary);
            assert!(summary.steps_processed > 0);
        }
    }
}
