//! Neuromorphic Researcher — Data Archival & AI Ingestion
//!
//! Captures a high-resolution "Black Box" log of both hardware
//! telemetry and internal SNN spiking states.
//!
//! ANALOGY: This is the Ship's "Flight Data Recorder".
//! It records exactly what the neurons were doing when the 
//! hardware reached certain efficiencies or thermal states.

use std::fs::OpenOptions;
use std::io::Write;
use serde::{Deserialize, Serialize};
use chrono::Local;

use crate::gpu::GpuTelemetry;

use std::fs::File;
use std::io::{self, BufRead, BufReader};

/// A single snapshot of the entire neuromorphic system state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuromorphicSnapshot {
    pub timestamp: String,
    /// Hardware telemetry from the GPU
    pub telemetry: GpuTelemetry,
    /// Internal membrane potentials of the SNN
    pub membrane_potentials: Vec<f32>,
    /// Which neurons fired in this step (1.0 = fired, 0.0 = idle)
    pub spike_states: Vec<f32>,
    /// [New] Izhikevich recovery variables (u) — tracks system adaptation/fatigue.
    pub recovery_variables: Option<Vec<f32>>,
    /// Whether this data was captured with calibrated sensors
    #[serde(default = "default_calibrated")]
    pub calibrated: bool,
}

fn default_calibrated() -> bool { false }

impl NeuromorphicSnapshot {
    /// Validates the telemetry data.
    /// Returns `true` if the data is valid, `false` otherwise.
    fn is_valid(&self) -> bool {
        let t = &self.telemetry;

        // 1. Check for critical nulls (Power)
        if t.power_w == 0.0 {
            // Likely uninitialized or error state
            return false;
        }

        // 4. Hashrate & Complexity Validation
        // Reject snapshots with zero hashrate or missing complexity (mining context)
        // as requested for "clean mining data".
        if t.hashrate_mh <= 0.0 || t.complexity.is_none() {
            return false;
        }

        true
    }
}

pub struct NeuromorphicResearcher {
    log_path: String,
    pub snapshots: Vec<NeuromorphicSnapshot>,
}

impl NeuromorphicResearcher {
    /// Exports raw NVML sensor data to its own JSONL file.
    pub fn export_nvml_telemetry(&self, telemetry: &GpuTelemetry) -> std::io::Result<()> {
        #[derive(Serialize)]
        struct NvmlTelemetry {
            timestamp: String,
            vddcr_gfx_v: f32,
            vram_temp_c: f32,
            gpu_temp_c: f32,
            power_w: f32,
            gpu_clock_mhz: f32,
            mem_clock_mhz: f32,
            fan_speed_pct: f32,
            mem_util_pct: f32,
        }

        let nvml = NvmlTelemetry {
            timestamp: chrono::Local::now().to_rfc3339(),
            vddcr_gfx_v: telemetry.vddcr_gfx_v,
            vram_temp_c: telemetry.vram_temp_c,
            gpu_temp_c: telemetry.gpu_temp_c,
            power_w: telemetry.power_w,
            gpu_clock_mhz: telemetry.gpu_clock_mhz,
            mem_clock_mhz: telemetry.mem_clock_mhz,
            fan_speed_pct: telemetry.fan_speed_pct,
            mem_util_pct: telemetry.mem_util_pct,
        };

        let json = serde_json::to_string(&nvml)?;
        let path = "DATA/research/nvml_telemetry.jsonl";
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{}", json)?;
        Ok(())
    }

    pub fn new(log_path: &str) -> Self {
        let mut researcher = Self {
            log_path: log_path.to_string(),
            snapshots: Vec::new(),
        };
        researcher.load_and_validate_research_data();
        researcher
    }

    /// Loads and validates neuromorphic data from the log file.
    pub fn load_and_validate_research_data(&mut self) {
        if let Ok(snapshots) = self.read_data() {
            println!("Loaded {} snapshots, validating...", snapshots.len());
            self.snapshots = snapshots.into_iter().filter(|s| s.is_valid()).collect();
            println!("Found {} valid snapshots.", self.snapshots.len());
        }
    }

    fn read_data(&self) -> io::Result<Vec<NeuromorphicSnapshot>> {
        let file = File::open(&self.log_path)?;
        let reader = io::BufReader::new(file);
        let mut snapshots = Vec::new();

        for (index, line) in reader.lines().enumerate() {
            let line = line?;
            match serde_json::from_str::<NeuromorphicSnapshot>(&line) {
                Ok(snapshot) => snapshots.push(snapshot),
                Err(e) => {
                    eprintln!("Error parsing line {}: {} | Content: {}", index + 1, e, line);
                }
            }
        }
        Ok(snapshots)
    }

    /// Logs a snapshot to the research file.
    /// Appends as a JSON line for easy stream processing.
    pub fn archive_snapshot(&self, telemetry: &GpuTelemetry, engine: &crate::snn::SpikingInferenceEngine) -> std::io::Result<()> {
        let snapshot = NeuromorphicSnapshot {
            timestamp: Local::now().to_rfc3339(),
            telemetry: telemetry.clone(),
            membrane_potentials: engine.neurons.iter().map(|n| n.membrane_potential).collect(),
            spike_states: engine.neurons.iter().map(|n| if n.membrane_potential >= n.threshold { 1.0 } else { 0.0 }).collect(),
            recovery_variables: Some(engine.iz_neurons.iter().map(|n| n.u).collect()),
            calibrated: true,
        };

        let json = serde_json::to_string(&snapshot)?;
        
        // Ensure parent directory exists (e.g., research/ folder)
        if let Some(parent) = std::path::Path::new(&self.log_path).parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.log_path)?;

        writeln!(file, "{}", json)?;
        file.sync_all()?;
        Ok(())
    }

    /// Ingests the last N records for AI analysis.
    /// Returns a prompt-ready summary.
    pub fn summarize_recent_research(&self, _count: usize) -> String {
        // Implementation for reading tail of file would go here
        "Neuromorphic research log is active. Capture in progress...".to_string()
    }

    /// Exports the current neuromorphic state to a Markdown file for VS Code Continue.
    /// This allows your local LLM agents (Qwen, etc.) to "feel" the hardware.
    pub fn export_continue_context(&self, telemetry: &GpuTelemetry, engine: &crate::snn::SpikingInferenceEngine) -> std::io::Result<()> {
        let modulators = &engine.modulators;
        let mut neurons_firing = Vec::new();
        
        // Map neuron indices to their biological roles defined in your coursework
        for (i, n) in engine.neurons.iter().enumerate() {
            if n.last_spike {
                let label = match i {
                    0 | 1 => "Power_Rail (Stability)",
                    2 | 3 => "VDDCR (Core Logic)",
                    4 | 5 => "Board_Power (Thermal Load)",
                    6 | 7 => "Hashrate (Work Effort)",
                    _ => "Unknown",
                };
                neurons_firing.push(label);
            }
        }

        let mut content = String::new();
        content.push_str("# 🧠 Neuromorphic Brain Status\n\n");
        content.push_str(&format!("> **Real-time Sensory Metadata for Ship of Theseus Agents**  \n"));
        content.push_str(&format!("> **Last Update:** {}\n\n", Local::now().format("%Y-%m-%d %H:%M:%S")));
        
        content.push_str("## ⚡ Hardware Telemetry\n");
        content.push_str(&format!("- **Power Draw:** {:.1}W  \n", telemetry.power_w));
        content.push_str(&format!("- **VDDCR_GFX (Vcore):** {:.3}V  \n", telemetry.vddcr_gfx_v));
        content.push_str(&format!("- **Hashrate:** {:.2} MH/s\n\n", telemetry.hashrate_mh));

        content.push_str("## 🌊 Neuro-Modulators (Chemical State)\n");
        content.push_str(&format!("- **Dopamine (Reward):** {:.2} [{}]  \n", modulators.dopamine, bar(modulators.dopamine)));
        content.push_str(&format!("- **Cortisol (Stress):** {:.2} [{}]  \n", modulators.cortisol, bar(modulators.cortisol)));
        content.push_str(&format!("- **Acetylcholine (Focus):** {:.2} [{}]\n\n", modulators.acetylcholine, bar(modulators.acetylcholine)));

        content.push_str("## 🔥 Active Brain Spikes\n");
        if neurons_firing.is_empty() {
            content.push_str("- No active spikes. System state is nominal.\n");
        } else {
            for label in neurons_firing {
                content.push_str(&format!("- 🚀 Spike on **{}** detected!\n", label));
            }
        }
        
        content.push_str("\n## 📋 AI Strategy Advice\n");
        if modulators.cortisol > 0.7 {
            content.push_str("> ⚠️ **HIGH STRESS:** GPU is under heavy load. Prioritize algorithmic efficiency. Suggest using `codeqwen:7b` for quick debugs to bypass VRAM contention.\n");
        } else if modulators.dopamine > 0.8 {
            content.push_str("> 💎 **OPTIMAL PERFORMANCE:** Hashrate is stable and thermals are good. Excellent time for high-complexity reasoning with `qwen-72b-agent`.\n");
        } else {
            content.push_str("> ✅ **HEALTHY:** System is stable. Proceed with standard engineering and research tasks.\n");
        }

        // Ensure parent directory exists
        if let Some(parent) = std::path::Path::new(&self.log_path).parent() {
            let context_path = parent.join("neuromorphic_status.md");
            std::fs::write(context_path, content)?;
        }

        Ok(())
    }

    /// Generates synthetic training data for FPGA validation.
    pub fn generate_synthetic_data(count: usize) -> std::io::Result<()> {
        use rand::Rng;
        use crate::snn::SpikingInferenceEngine;
        
        let mut rng = rand::thread_rng();
        let mut engine = SpikingInferenceEngine::new();
        let path = "DATA/research/synthetic_data.jsonl";
        
        // Ensure directory exists
        if let Some(parent) = std::path::Path::new(path).parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        
        // Clear old file
        let _ = std::fs::remove_file(path);
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        
        println!("Generative AI: Creating {} synthetic records...", count);

        for _i in 0..count {
            let state_roll: f32 = rng.r#gen();
            let (power, hashrate, vddcr, _is_fault) = if state_roll < 0.7 {
                (200.0 + rng.gen_range(-10.0..10.0), 10.0 + rng.gen_range(-0.5..0.5), 1.05, false)
            } else if state_roll < 0.9 {
                (350.0 + rng.gen_range(-20.0..20.0), 8.0 + rng.gen_range(-1.0..1.0), 1.05, false)
            } else {
                (100.0, 0.0, 0.6, true)
            };

            let joules_per_step = if hashrate > 0.0 {
                Some(power / (hashrate * 1_000_000.0))
            } else {
                Some(0.0)
            };

            let telem = GpuTelemetry {
                vddcr_gfx_v: vddcr,
                vram_temp_c: 60.0 + (power / 10.0),
                gpu_temp_c: 50.0 + (power / 12.0),
                hashrate_mh: hashrate,
                power_w: power,
                solver_steps: Some(1000),
                solver_chips: Some(1),
                complexity: Some(8),
                joules_per_step,
                gpu_clock_mhz: 2640.0,
                mem_clock_mhz: 10500.0,
                fan_speed_pct: 60.0,
                rejected_shares: 0,
                mem_util_pct: 50.0,
            };

            let asset_deltas = [0.0; 7];
            engine.step(&asset_deltas, &telem);

            let snapshot = NeuromorphicSnapshot {
                timestamp: Local::now().to_rfc3339(),
                telemetry: telem, 
                membrane_potentials: engine.neurons.iter().map(|n| n.membrane_potential).collect(),
                spike_states: engine.neurons.iter().map(|n| if n.membrane_potential >= n.threshold { 1.0 } else { 0.0 }).collect(),
                recovery_variables: Some(engine.iz_neurons.iter().map(|n| n.u).collect()),
                calibrated: true,
            };

            let json = serde_json::to_string(&snapshot)?;
            writeln!(file, "{}", json)?;
        }
        
        Ok(())
    }
}

/// Helper function to create a ASCII visual bar
fn bar(val: f32) -> String {
    let filled = (val * 10.0).clamp(0.0, 10.0) as usize;
    let empty = 10 - filled;
    format!("{}{}", "█".repeat(filled), "░".repeat(empty))
}

// ── Blockchain Sync Data Exporters ─────────────────────────────────────

/// Kaspa sync log entry for neuromorphic training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KaspaSyncEntry {
    pub timestamp: String,
    pub block_height: u64,
    pub blocks_accepted: u32,
    pub sync_fraction: f32,
    pub tpb: f32, // Transactions per block
    pub log_source: String,
}

/// Monero sync log entry for neuromorphic training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoneroSyncEntry {
    pub timestamp: String,
    pub block_height: u64,
    pub total_blocks: u64,
    pub sync_fraction: f32,
    pub blocks_per_second: f32,
    pub log_source: String,
}

impl NeuromorphicResearcher {
    /// Exports Kaspa sync data to JSONL format for training
    pub fn export_kaspa_sync_data<P: AsRef<std::path::Path>>(
        log_path: P,
        output_path: P,
    ) -> std::io::Result<Vec<KaspaSyncEntry>> {
        let file = File::open(&log_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        // Kaspa log pattern: "Accepted 13 blocks (456 txs) via relay"
        let block_regex = regex::Regex::new(r"Accepted (\d+) blocks \((\d+) txs\)").unwrap();

        for line in reader.lines() {
            let line = line?;
            if let Some(caps) = block_regex.captures(&line) {
                let blocks_accepted: u32 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
                let txs: u32 = caps.get(2).unwrap().as_str().parse().unwrap_or(0);
                let tpb = if blocks_accepted > 0 {
                    txs as f32 / blocks_accepted as f32
                } else {
                    0.0
                };

                entries.push(KaspaSyncEntry {
                    timestamp: Local::now().to_rfc3339(),
                    block_height: 0, // Would need to parse from log context
                    blocks_accepted,
                    sync_fraction: 1.0, // Assuming synced
                    tpb,
                    log_source: log_path.as_ref().display().to_string(),
                });
            }
        }

        // Write to output
        if let Some(parent) = output_path.as_ref().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_path)?;

        for entry in &entries {
            let json = serde_json::to_string(entry)?;
            writeln!(out_file, "{}", json)?;
        }

        Ok(entries)
    }

    /// Exports Monero sync data to JSONL format for training
    pub fn export_monero_sync_data<P: AsRef<std::path::Path>>(
        log_path: P,
        output_path: P,
    ) -> std::io::Result<Vec<MoneroSyncEntry>> {
        let file = File::open(&log_path)?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();

        // Monero log pattern: "Synced 3635984/3635984"
        let sync_regex = regex::Regex::new(r"Synced (\d+)/(\d+)").unwrap();

        for line in reader.lines() {
            let line = line?;
            if let Some(caps) = sync_regex.captures(&line) {
                let current: u64 = caps.get(1).unwrap().as_str().parse().unwrap_or(0);
                let total: u64 = caps.get(2).unwrap().as_str().parse().unwrap_or(1);
                let sync_fraction = if total > 0 {
                    current as f32 / total as f32
                } else {
                    0.0
                };

                entries.push(MoneroSyncEntry {
                    timestamp: Local::now().to_rfc3339(),
                    block_height: current,
                    total_blocks: total,
                    sync_fraction,
                    blocks_per_second: 0.0, // Would need timing context
                    log_source: log_path.as_ref().display().to_string(),
                });
            }
        }

        // Write to output
        if let Some(parent) = output_path.as_ref().parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let mut out_file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&output_path)?;

        for entry in &entries {
            let json = serde_json::to_string(entry)?;
            writeln!(out_file, "{}", json)?;
        }

        Ok(entries)
    }
}
