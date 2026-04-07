use spikenaut_fpga_supervisor::gpu::{HardwareBridge, GpuTelemetry};
use spikenaut_fpga_supervisor::fpga::FpgaBridge;
use spikenaut_fpga_supervisor::snn::SpikingInferenceEngine;
use spikenaut_fpga_supervisor::research::NeuromorphicResearcher;
use spikenaut_fpga_supervisor::cpu;
use std::time::Duration;
use std::io::{self, Write};
use tokio::time::sleep;

/// SPIKENAUT-FPGA-SUPERVISOR - v1.0.0
/// Standalone Hardware Orchestration & Neuromorphic Telemetry
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ── Telemetry Initialization ────────────────────────────────
    cpu::init_telemetry();
    tokio::spawn(cpu::run_metrics_collector());

    // ── Instance Protection (Lockfile) ───────────────────────────
    struct LockGuard(&'static str);
    impl Drop for LockGuard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(self.0);
        }
    }

    let lock_path = "/tmp/spikenaut_supervisor.lock";
    
    if std::path::Path::new(lock_path).exists() {
        if let Ok(content) = std::fs::read_to_string(lock_path) {
            if let Ok(old_pid) = content.trim().parse::<u32>() {
                if std::path::Path::new(&format!("/proc/{}", old_pid)).exists() {
                    eprintln!("[supervisor] FATAL: Another instance is already active (PID: {}).", old_pid);
                    std::process::exit(1);
                }
            }
        }
    }
    
    std::fs::write(lock_path, std::process::id().to_string())?;
    let _lock_guard = LockGuard(lock_path);

    println!("[supervisor] --- Spikenaut FPGA Supervisor ---");
    
    // Auto-detect port
    let ports = ["/dev/ttyUSB0", "/dev/ttyUSB1", "/dev/ttyUSB2"];
    let mut bridge = None;

    for p in &ports {
        match FpgaBridge::new(p) {
            Ok(b) => {
                println!("[supervisor] FPGA connected on {}", p);
                bridge = Some(b);
                break;
            }
            Err(_) => {}
        }
    }

    let mut bridge: FpgaBridge = match bridge {
        Some(b) => b,
        None => {
            eprintln!("[supervisor] FATAL: No functional Artix-7 serial ports found.");
            std::process::exit(1);
        }
    };

    // ── Neuromorphic Context Engines ─────────────────────────────
    let mut engine = SpikingInferenceEngine::new();
    let research_path = "DATA/research/neuromorphic_data.jsonl";
    let model_path = "DATA/research/student_model.json";

    if std::path::Path::new(model_path).exists() {
        println!("[supervisor] Loading brain state...");
        let _ = engine.load_parameters(model_path);
    }

    let researcher = NeuromorphicResearcher::new(research_path);
    let mut step_count = 0;

    // ── IPC Server ───────────────────────────────────────────────
    let udp_socket = std::net::UdpSocket::bind("127.0.0.1:9898").expect("FATAL: Failed to bind IPC socket");
    udp_socket.set_nonblocking(true).expect("FATAL: Failed to set non-blocking UDP");

    loop {
        step_count += 1;
        let telem = HardwareBridge::read_telemetry();
        
        // --- SiliconBridge v3.0: Asset Market Sentiment ---
        // Assets: DNX, QUAI, QUBIC, KAS, XMR, OCEAN, VERUS
        let mut rng = rand::thread_rng();
        let mut asset_deltas = [0.0f32; 7];
        for d in &mut asset_deltas {
            *d = rng.gen_range(-0.05..0.05); // Simulated price delta
        }

        // Map Telemetry to Stimuli
        let stimuli = encode_telemetry(&asset_deltas, &telem, &engine);

        // ... rest of UDP logic

        // Process IPC
        let mut buf = [0; 4096];
        while let Ok((amt, src)) = udp_socket.recv_from(&mut buf) {
            if let Ok(msg) = std::str::from_utf8(&buf[..amt]) {
                if let Ok(json) = serde_json::from_str::<serde_json::Value>(msg) {
                    match json["type"].as_str() {
                        Some("LearningReward") => {
                            let dopamine = json["dopamine_delta"].as_f64().unwrap_or(0.0) as f32;
                            let cortisol = json["cortisol_delta"].as_f64().unwrap_or(0.0) as f32;
                            engine.inject_learning_reward(dopamine, cortisol);
                        }
                        Some("GetNeuroState") => {
                            let state_json = serde_json::json!({
                                "dopamine": engine.modulators.dopamine,
                                "cortisol": engine.modulators.cortisol,
                                "acetylcholine": engine.modulators.acetylcholine,
                                "lif_spike_count": engine.neurons.iter().filter(|n| n.last_spike).count()
                            });
                            if let Ok(encoded) = serde_json::to_string(&state_json) {
                                let _ = udp_socket.send_to(encoded.as_bytes(), src);
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        engine.step(&asset_deltas, &telem);
        let _ = researcher.archive_snapshot(&telem, &engine);
        
        if step_count % 10 == 0 {
            let _ = researcher.export_continue_context(&telem, &engine);
        }

        match bridge.step_cluster(&stimuli) {
            Ok((potentials, spikes)) => {
                if spikes == 0xFFFF {
                    println!("\n[!] EMERGENCY BRAKE TRIGGERED");
                    let _ = HardwareBridge::apply_emergency_brake(0.6);
                }
                print_dashboard(&telem, &potentials, spikes, step_count);
            }
            Err(e) => {
                eprintln!("\n[supervisor] FPGA Sync Lost: {}", e);
                break;
            }
        }

        sleep(Duration::from_millis(100)).await; 
    }
    Ok(())
}

fn encode_telemetry(deltas: &[f32; 7], _t: &GpuTelemetry, engine: &SpikingInferenceEngine) -> [u16; spikenaut_fpga_supervisor::NUM_NEURONS] {
    let mut stims = [0u16; spikenaut_fpga_supervisor::NUM_NEURONS];
    
    // SiliconBridge v3.0: Map Asset Sentiment to Trading Core (N0-N13)
    for i in 0..7 {
        let delta = deltas[i];
        let bear_stim = (-delta).max(0.0).clamp(0.0, 1.0);
        let bull_stim = delta.max(0.0).clamp(0.0, 1.0);
        
        // Odd = Bull (+), Even = Bear (-)
        // Q8.8 Format: value * 256
        stims[i * 2] = (bear_stim * 256.0) as u16;
        stims[i * 2 + 1] = (bull_stim * 256.0) as u16;
    }
    
    // N14/N15 are regulatory and driven internally by the engine/FPGA feedback
    stims[14] = 0;
    stims[15] = 0;
    
    stims
}

fn print_dashboard(t: &GpuTelemetry, pots: &[u16], spikes: u16, step: u64) {
    print!("\r[Step {}] Pwr: {:5.1}W | Vcore: {:.3}V | Spikes: {:04X} | Pot0: {:04X} ", 
           step, t.power_w, t.vddcr_gfx_v, spikes, pots[0]);
    let _ = io::stdout().flush();
}
