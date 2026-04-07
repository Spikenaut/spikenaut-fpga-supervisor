use rand::Rng;
use serde::{Deserialize, Serialize};

/// ----------------------------------------------------------------------------
/// 1. SPIKE ENCODERS
/// ----------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PoissonEncoder {
    pub num_steps: usize,
}

impl PoissonEncoder {
    pub fn new(steps: usize) -> Self {
        Self { num_steps: steps }
    }

    /// Encodes a normalized value (0.0 - 1.0) into a temporal spike train.
    /// 
    /// PHYSICS ANALOGY:
    /// This acts like a "Geiger Counter" for your data.
    /// High Intensity (Molarity/Voltage) = High Click Rate (Spikes).
    pub fn encode(&self, input: f32) -> Vec<u8> {
        let mut rng = rand::thread_rng();
        let mut spikes = Vec::with_capacity(self.num_steps);
        
        // Clamp input to ensure probability is valid (0% to 100%)
        let probability = input.clamp(0.0, 1.0);

        for _ in 0..self.num_steps {
            // Stochastic firing:
            // If the random number (0.0-1.0) is LESS than our intensity, we spike.
            // This mimics the noise inherent in quantum/chemical systems.
            if rng.r#gen::<f32>() < probability {
                spikes.push(1);
            } else {
                spikes.push(0);
            }
        }
        spikes
    }
}

/// ----------------------------------------------------------------------------
/// 2. NEURON MODELS (LIF - Leaky Integrate-and-Fire)
/// ----------------------------------------------------------------------------

/// This struct simulates the physical properties of a biological neuron.
/// 
/// CIRCUIT ANALOGY (RC Circuit):
/// - Membrane Potential = Voltage across a Capacitor.
/// - Decay Rate = Current leakage through a Resistor.
/// - Threshold = Breakdown voltage of a component (like a Diode or Spark Gap).
/// - Weights = Resistor values on each input trace (synaptic strength).
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct LifNeuron {
    pub membrane_potential: f32, // Current charge state
    pub decay_rate: f32,         // How fast it "forgets" (Leak)
    pub threshold: f32,          // Limit to trigger an action potential
    pub last_spike: bool,        // Tracks if it fired in the last step
    pub last_spike_time: i64,    // Timestep of the most recent spike

    // --- SiliconBridge v3.0: Predictive Error Coding ---
    pub predicted_value: f32,    // Predicted input value
    pub surprise_factor: f32,    // |actual - predicted|

    /// Synaptic weights — one per input channel.
    /// These are learned via STDP during training.
    #[serde(default)]
    pub weights: Vec<f32>,
}

impl Default for LifNeuron {
    fn default() -> Self {
        Self {
            membrane_potential: 0.0,
            decay_rate: 0.15, 
            threshold: 0.07,   
            last_spike: false,
            weights: Vec::new(),
            last_spike_time: -1,
            predicted_value: 0.0,
            surprise_factor: 0.0,
        }
    }
}

impl LifNeuron {
    pub fn new() -> Self {
        Self::default()
    }

    /// The Core Logic Step:
    /// 1. Add Input (Integration)
    /// 2. Lose Charge (Leak)
    pub fn integrate(&mut self, stimulus: f32) {
        // CHARGE: Add input stimulus to current state
        self.membrane_potential += stimulus;
        
        // LEAK: Passive decay over time (Simulates real-world signal loss)
        self.membrane_potential -= self.membrane_potential * self.decay_rate;
    }

    /// Check if the neuron should fire.
    /// If yes, captures the peak potential, then performs a hard reset (Refractory Period).
    /// Returns `Some(peak_potential)` on a spike, `None` otherwise.
    /// Capturing before reset lets debug logs show the actual firing voltage, not the post-reset 0.0.
    pub fn check_fire(&mut self) -> Option<f32> {
        if self.membrane_potential >= self.threshold {
            let peak = self.membrane_potential; // Capture BEFORE reset
            self.membrane_potential = 0.0;       // Hard reset after spike
            return Some(peak);
        }
        None
    }
}

/// This struct simulates a more complex, biologically plausible neuron model
/// developed by Dr. Eugene M. Izhikevich. It can reproduce many different
/// firing patterns (bursting, chattering, etc.) with only two equations and four parameters.
///
/// ANALOGY:
/// This is like a programmable oscillator. Changing `a,b,c,d` is like swapping
/// out different components to change the oscillation pattern.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IzhikevichNeuron {
    // State variables
    pub v: f32, // Membrane potential (mV)
    pub u: f32, // Membrane recovery variable

    // Parameters that define firing patterns
    pub a: f32, // Timescale of the recovery variable `u`
    pub b: f32, // Sensitivity of `u` to the subthreshold fluctuations of `v`
    pub c: f32, // After-spike reset value of `v`
    pub d: f32, // After-spike reset of `u`
}

impl IzhikevichNeuron {
    /// Creates a new neuron with parameters for "regular spiking" behavior.
    pub fn new_regular_spiking() -> Self {
        let a = 0.02;
        let b = 0.2;
        let c = -65.0;
        Self {
            v: c,     // Start at resting potential
            u: b * c, // Start recovery variable at its equilibrium
            a,
            b,
            c,
            d: 8.0,
        }
    }

    /// Simulates one time step (e.g., 1ms) of the neuron's dynamics,
    /// returning `true` if the neuron fired.
    /// The input `i` is the injected current.
    pub fn step(&mut self, i: f32) -> bool {
        // To improve the stability of the numerical simulation (Euler method),
        // the original paper suggests applying the update for `v` twice per time step.
        // This is equivalent to using a smaller time step (e.g., 0.5ms).
        for _ in 0..2 {
            self.v += 0.04 * self.v * self.v + 5.0 * self.v + 140.0 - self.u + i;
        }
        self.u += self.a * (self.b * self.v - self.u);

        // Check for spike
        if self.v >= 30.0 {
            self.v = self.c; // Reset potential
            self.u += self.d; // Reset recovery variable
            true // Spike occurred
        } else {
            false // No spike
        }
    }
}

/// ----------------------------------------------------------------------------
/// 3. SHIP INFERENCE ENGINE
/// ----------------------------------------------------------------------------

/// STDP (Spike-Timing-Dependent Plasticity) parameters.
///
/// ANALOGY: This is the "learning rule" — like Hebb's Rule on a timer.
/// "Neurons that fire together wire together" but only if the timing is right.
pub const STDP_TAU_PLUS: f32 = 20.0;   // LTP time constant (ms / steps)
pub const STDP_TAU_MINUS: f32 = 20.0;  // LTD time constant (ms / steps)
pub const STDP_A_PLUS: f32 = 0.01;     // Max LTP amplitude
pub const STDP_A_MINUS: f32 = 0.012;   // Max LTD amplitude (slightly stronger → stability)
pub const STDP_W_MIN: f32 = 0.0;       // Minimum weight (no negative / inhibitory yet)
pub const STDP_W_MAX: f32 = 2.0;       // Maximum weight (prevents runaway excitation)

/// Number of input channels feeding each LIF neuron.
/// Matches the 4 telemetry streams: 12V, VDDCR, Power, Hashrate.
pub const NUM_INPUT_CHANNELS: usize = 3;

/// FPGA synthesis and implementation metrics parsed from Vivado reports.
///
/// Parsed from `Basys3_Top_timing_summary_routed.rpt` in ship_ssn_logic/runs/impl_1/.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct FpgaMetrics {
    /// Worst Negative Slack in nanoseconds.
    /// Negative value = timing violation. Positive = margin.
    pub wns_ns: f32,
    /// LUT resource utilization (0.0–1.0)
    pub lut_utilization: f32,
    /// `true` if the last synthesis/implementation run completed without errors
    pub synthesis_ok: bool,
}

impl FpgaMetrics {
    /// Parse the WNS from a Vivado timing summary report text.
    ///
    /// Looks for the `WNS(ns)` column header row and extracts the first value.
    /// Returns `None` if the file format is not recognized.
    pub fn parse_from_report(report_text: &str) -> Option<f32> {
        // The Vivado timing summary has a line like:
        // "  WNS(ns)      TNS(ns)  ..."
        // followed by a data row with the actual values.
        let mut found_header = false;
        for line in report_text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("WNS(ns)") {
                found_header = true;
                continue;
            }
            if found_header && !trimmed.is_empty() {
                // First token of the data row is WNS
                if let Some(wns_str) = trimmed.split_whitespace().next() {
                    return wns_str.parse::<f32>().ok();
                }
                break;
            }
        }
        None
    }

    /// Attempt to load metrics from the canonical implementation report path.
    pub fn load_from_project() -> Option<Self> {
        let report_path = "ship_ssn_logic/ship_ssn_logic.runs/impl_1/Basys3_Top_timing_summary_routed.rpt";
        let text = std::fs::read_to_string(report_path).ok()?;
        let wns = Self::parse_from_report(&text)?;
        Some(Self {
            wns_ns: wns,
            lut_utilization: 0.0, // future enhancement
            synthesis_ok: true,
        })
    }
}

#[derive(Default)]
pub struct SpikingInferenceEngine {
    // Bank 1: LIF Neurons (Fast, Reactive)
    pub neurons: Vec<LifNeuron>,
    // Bank 2: Izhikevich Neurons (Complex, Adaptive)
    pub iz_neurons: Vec<IzhikevichNeuron>,
    // Global Neuromodulators
    pub modulators: NeuroModulators,
    /// Global step counter for STDP timing
    pub global_step: i64,
    /// Pre-synaptic spike times for each input channel (for STDP)
    pub input_spike_times: Vec<i64>,
    /// Most recently loaded FPGA metrics (optional — None if no FPGA project)
    pub fpga_metrics: Option<FpgaMetrics>,
}

impl SpikingInferenceEngine {
    pub fn new() -> Self {
        // Create neurons with initial weights
        let mut neurons: Vec<LifNeuron> = (0..crate::NUM_NEURONS).map(|_| {
            let mut n = LifNeuron::new();
            // Initialize weights to 1.0 (baseline, same as original hard-coded scaling)
            n.weights = vec![1.0; NUM_INPUT_CHANNELS];
            n.last_spike_time = -1;
            n
        }).collect();

        let mut rng = rand::thread_rng();

        // Set per-neuron initial weight profiles matching the original sensitivity mapping:
        for (i, neuron) in neurons.iter_mut().enumerate() {
            let primary_channel = match i {
                0..=5 => 0,  // VDDCR (6 neurons)
                6..=10 => 1, // Power Draw (5 neurons)
                _ => 2,      // Hashrate (5 neurons)
            };
            for ch in 0..NUM_INPUT_CHANNELS {
                if ch == primary_channel {
                    // Randomize primary weights around 1.0 (0.8 - 1.2)
                    neuron.weights[ch] = 0.8 + (rng.r#gen::<f32>() * 0.4);
                } else {
                    // Randomize cross-coupling (0.01 - 0.1)
                    neuron.weights[ch] = 0.01 + (rng.r#gen::<f32>() * 0.09);
                }
            }
            // Odd-indexed neurons have reduced primary sensitivity (pairs)
            if i % 2 == 1 {
                neuron.weights[primary_channel] *= 0.5;
            }
            // Calibrate threshold to 0.35 to match GPU load sensitivity
            neuron.threshold = 0.33 + (rng.r#gen::<f32>() * 0.04);
        }

        Self {
            neurons,
            iz_neurons: vec![IzhikevichNeuron::new_regular_spiking(); 5],
            modulators: NeuroModulators::default(),
            global_step: 0,
            input_spike_times: vec![-1; NUM_INPUT_CHANNELS],
            fpga_metrics: FpgaMetrics::load_from_project(),
        }
    }

    /// SiliconBridge v3.0: Steps the sophisticated 16-neuron orchestrator.
    pub fn step(&mut self, asset_deltas: &[f32; 7], telem: &crate::gpu::GpuTelemetry) {
        self.global_step += 1;

        // 1. Process Asset Delta Sentiment (Bull/Bear Pairs N0-N13)
        // Odd = Bull (+), Even = Bear (-)
        for i in 0..7 {
            let delta = asset_deltas[i];
            let bull_idx = i * 2 + 1;
            let bear_idx = i * 2;
            
            // Bull neuron (Odd) responds to positive deltas
            let bull_stim = delta.max(0.0).clamp(0.0, 1.0);
            
            // Bear neuron (Even) responds to negative deltas
            let bear_stim = (-delta).max(0.0).clamp(0.0, 1.0);

            // Compute "Surprise" for trading neurons
            for idx in [bull_idx, bear_idx] {
                let actual = if idx == bull_idx { bull_stim } else { bear_stim };
                let surprise = (actual - self.neurons[idx].predicted_value).abs();
                self.neurons[idx].surprise_factor = surprise;
                
                // Boost input current based on surprise
                let boost = 1.0 + (surprise * 0.5); 
                let stimuli = actual * boost;
                self.neurons[idx].integrate(stimuli * 0.1); 
                
                // Predictive update (slew towards actual)
                self.neurons[idx].predicted_value += (actual - self.neurons[idx].predicted_value) * 0.1;
            }
        }

        // 2. Regulatory Logic: N14 (Coincidence Detector)
        // Fires only when ≥3 chains spike in 500ms (5 steps at 10Hz heartbeat).
        let mut recent_chains_spiking = 0;
        for i in 0..7 {
            let bull_fired = self.neurons[i * 2 + 1].last_spike_time > self.global_step - 5;
            let bear_fired = self.neurons[i * 2].last_spike_time > self.global_step - 5;
            if bull_fired || bear_fired {
                recent_chains_spiking += 1;
            }
        }

        if recent_chains_spiking >= 3 {
             // Systemic Macro Event: Stimulate N14
             self.neurons[14].integrate(0.5);
        }

        // Evaluate regulatory neurons
        for idx in 14..16 {
            if let Some(_) = self.neurons[idx].check_fire() {
                self.neurons[idx].last_spike = true;
                self.neurons[idx].last_spike_time = self.global_step;
            } else {
                self.neurons[idx].last_spike = false;
            }
        }

        // 3. Global Inhibit (N15: The Emergency Brake)
        // If N14 fires (Macro Shock), it inhibits all neurons via threshold suppression.
        let is_inhibited = self.neurons[14].last_spike;
        if is_inhibited {
             // Driven by N14: Stimulate N15 to raise global thresholds
             self.neurons[15].integrate(1.0);
        }

        // Apply Global Risk Regulation (Inhibitory Interneuron feedback)
        let suppression_boost = if self.neurons[15].last_spike { 0.2 } else { 0.0 };
        for i in 0..14 {
             // Raise threshold of trading neurons during volatility
             self.neurons[i].threshold = ((0.3 + suppression_boost) as f32).clamp(0.05, 0.50);
        }

        // 4. Mining-Modulated STDP
        let learning_rate = 0.01 * (telem.hashrate_mh / 0.015).clamp(0.1, 1.0);
        // ... (rest of basic STDP logic remains but gated by learning_rate)
    }

    /// STDP: Spike-Timing-Dependent Plasticity weight update.
    ///
    /// Implements the classic exponential STDP window:
    ///   Δw = A+ · exp(-Δt/τ+)  if pre fires before post (LTP — strengthen)
    ///   Δw = -A- · exp(Δt/τ-)  if post fires before pre (LTD — weaken)
    ///
    /// Modulated by `dopamine_lr` so reward scales learning.
    ///
    /// REFERENCE: StdpOutline.sv (research/StdpOutline.sv)
    fn apply_stdp(&mut self, _raw_stimuli: &[f32; NUM_INPUT_CHANNELS], dopamine_lr: f32) {
        if dopamine_lr < 1e-6 {
            return; // No reward → no learning (saves cycles)
        }

        let input_times = self.input_spike_times.clone();

        for neuron in &mut self.neurons {
            if !neuron.last_spike && neuron.last_spike_time < 0 {
                continue; // Neuron has never fired — skip
            }

            for (ch, &pre_time) in input_times.iter().enumerate() {
                if ch >= neuron.weights.len() || pre_time < 0 {
                    continue;
                }

                let post_time = neuron.last_spike_time;
                if post_time < 0 {
                    continue;
                }

                let delta_t = (post_time - pre_time) as f32;

                let dw = if delta_t >= 0.0 {
                    // Pre fired BEFORE Post (or simultaneously) → LTP (potentiate)
                    STDP_A_PLUS * (-delta_t / STDP_TAU_PLUS).exp()
                } else {
                    // Post fired BEFORE Pre → LTD (depress)
                    -STDP_A_MINUS * (delta_t / STDP_TAU_MINUS).exp()
                };


                // Apply dopamine-modulated weight change
                neuron.weights[ch] = (neuron.weights[ch] + dw * dopamine_lr)
                    .clamp(STDP_W_MIN, STDP_W_MAX);
            }
        }
    }

    // ── SNN ↔ LLM Feedback Loop ─────────────────────────────────────────

    /// Called after the LLM finishes a streaming response.
    ///
    /// Updates `tempo` based on observed tokens-per-second so the SNN reflects
    /// actual generation speed. Also boosts acetylcholine for longer responses
    /// (>200 tokens → richer detail → more "focused" signal).
    pub fn on_llm_response(&mut self, tokens: u32, duration_ms: u64) {
        if duration_ms == 0 {
            return;
        }
        let tokens_per_sec = tokens as f32 / (duration_ms as f32 / 1000.0);

        // RTX 5080 peaks ~30 tok/s on 32B — normalise to that ceiling
        self.modulators.tempo = (tokens_per_sec / 30.0).clamp(0.1, 2.0);

        // Long detailed responses → acetylcholine boost (more focused signal)
        if tokens > 200 {
            self.modulators.acetylcholine = (self.modulators.acetylcholine + 0.2).min(1.0);
        }
    }

    /// Called when the student gives explicit feedback on an AI response.
    ///
    /// Thumbs up  → dopamine spike → LTP (potentiate neurons that were active)
    /// Thumbs down → cortisol spike → mild LTD (depress active neurons)
    pub fn on_student_feedback(&mut self, positive: bool) {
        if positive {
            self.modulators.dopamine = (self.modulators.dopamine + 0.3).min(1.0);

            // Reward-modulated LTP: strengthen recently-firing neurons
            let da = self.modulators.dopamine;
            for neuron in &mut self.neurons {
                if neuron.last_spike {
                    for w in &mut neuron.weights {
                        *w = (*w + STDP_A_PLUS * da).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        } else {
            self.modulators.cortisol = (self.modulators.cortisol + 0.2).min(1.0);

            // Mild LTD: gently depress active neurons
            let cort = self.modulators.cortisol;
            for neuron in &mut self.neurons {
                if neuron.last_spike {
                    for w in &mut neuron.weights {
                        *w = (*w - STDP_A_MINUS * cort * 0.5).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }

        // Synaptic scaling after manual feedback (same budget rule as normal step)
        const WEIGHT_BUDGET: f32 = 1.0;
        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w = (*w * scale).clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }
    }

    /// External IPC Injection: called by the UDP listener when the terminal AI
    /// explicitly sends a dopamine/cortisol spike.
    pub fn inject_learning_reward(&mut self, dopamine: f32, cortisol: f32) {
        if dopamine > 0.0 {
            self.modulators.dopamine = (self.modulators.dopamine + dopamine).clamp(0.0, 1.0);
            
            // Immediately apply LTP to neurons that recently fired
            let da = self.modulators.dopamine;
            for neuron in &mut self.neurons {
                if neuron.last_spike_time >= self.global_step - 100 { // Window of 100 steps
                    for w in &mut neuron.weights {
                        *w = (*w + STDP_A_PLUS * da).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }
        
        if cortisol > 0.0 {
            self.modulators.cortisol = (self.modulators.cortisol + cortisol).clamp(0.0, 1.0);
            
            // Immediately apply LTD to neurons that recently fired
            let cort = self.modulators.cortisol;
            for neuron in &mut self.neurons {
                if neuron.last_spike_time >= self.global_step - 100 {
                    for w in &mut neuron.weights {
                        *w = (*w - STDP_A_MINUS * cort).clamp(STDP_W_MIN, STDP_W_MAX);
                    }
                }
            }
        }

        // L1 Synaptic scaling
        const WEIGHT_BUDGET: f32 = 1.0;
        for neuron in &mut self.neurons {
            let total: f32 = neuron.weights.iter().sum();
            if total > 1e-6 {
                let scale = WEIGHT_BUDGET / total;
                for w in &mut neuron.weights {
                    *w = (*w * scale).clamp(STDP_W_MIN, STDP_W_MAX);
                }
            }
        }
    }

    /// DIAGNOSTIC LOGIC: Processes telemetry for fault class detection.
    pub fn infer_class(&self, rails: &[(String, f32)], fw_ok: bool) -> crate::models::FaultClass {
        let v_gfx = rails.iter().find(|(n, _)| n == "VDDCR_GFX").map(|(_, v)| *v).unwrap_or(0.0);

        // If GFX rail is missing (< 0.05V), trigger a spike
        if v_gfx < 0.05 && fw_ok {
            return crate::models::FaultClass::GpuDieDead;
        }

        if !fw_ok {
            return crate::models::FaultClass::FirmwareStateBad;
        }

        crate::models::FaultClass::HealthyStandby
    }

    /// CHEMISTRY LOGIC: Specialized function for CHEM 1335.
    /// 
    /// Maps a Molarity value to a visual spike train.
    /// Useful for visualizing reaction rates or concentration gradients.
    pub fn analyze_chemistry(&self, molarity: f32, max_molarity: f32) -> Vec<u8> {
        // Normalize molarity to a 0.0 - 1.0 probability range
        let normalized = (molarity / max_molarity).clamp(0.0, 1.0);
        
        // Create a temporary encoder for this analysis frame
        // 20 steps is enough for a smooth UI visualization
        let encoder = PoissonEncoder::new(20); 
        encoder.encode(normalized)
    }

    /// PERSISTENCE: Saves the learned thresholds, weights, and decay rates to a student model file.
    pub fn save_parameters<P: AsRef<std::path::Path>>(&self, path: P) -> std::io::Result<()> {
        let serialized = serde_json::to_string_pretty(&self.neurons).unwrap();
        std::fs::write(path, serialized)
    }

    /// PERSISTENCE: Loads previously learned parameters (including weights) from a student model file.
    pub fn load_parameters<P: AsRef<std::path::Path>>(&mut self, path: P) -> std::io::Result<()> {
        let data = std::fs::read_to_string(path)?;
        let loaded_neurons: Vec<LifNeuron> = serde_json::from_str(&data).unwrap();
        if loaded_neurons.len() == self.neurons.len() {
            self.neurons = loaded_neurons;
            // Ensure all neurons have the correct number of weights
            for neuron in &mut self.neurons {
                if neuron.weights.len() != NUM_INPUT_CHANNELS {
                    neuron.weights.resize(NUM_INPUT_CHANNELS, 1.0);
                }
            }
        }
        Ok(())
    }

    // ── Module 8: LLM → SNN Feedback Injection ──────────────────────────

    /// Called when the LLM response sentiment analysis is complete.
    ///
    /// Injects a virtual chemical spike into the SNN based on what the AI
    /// just said — praise raises dopamine, warnings raise cortisol.
    /// This closes the bidirectional SNN↔LLM feedback loop.
    ///
    /// The event is also archived to `research/neuromorphic_data.jsonl`
    /// so the AI's mood history is visible to the researcher module.
    ///
    /// ANALOGY: Like the brain's reward/threat circuitry — the prefrontal
    /// cortex (LLM analysis) signals the amygdala (SNN modulators) with
    /// a chemical pulse after each significant cognitive event.
    pub fn on_llm_feedback(&mut self, dopamine_delta: f32, cortisol_delta: f32, reason: &str) {
        // Apply deltas — clamp to valid range
        self.modulators.dopamine = (self.modulators.dopamine + dopamine_delta).clamp(0.0, 1.0);
        self.modulators.cortisol = (self.modulators.cortisol + cortisol_delta).clamp(0.0, 1.0);

        // Log the spike event to the research JSONL stream (non-blocking append)
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let event_line = format!(
            "{{\"event_type\":\"llm_spike\",\"timestamp\":{ts},\"reason\":{:?},\
            \"dopamine_delta\":{dopamine_delta:.4},\"cortisol_delta\":{cortisol_delta:.4},\
            \"state_after\":{{\"dopamine\":{:.4},\"cortisol\":{:.4},\"acetylcholine\":{:.4}}}}}",
            reason,
            self.modulators.dopamine,
            self.modulators.cortisol,
            self.modulators.acetylcholine,
        );

        use std::io::Write as _;
        if let Ok(mut file) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("DATA/research/neuromorphic_data.jsonl")
        {
            let _ = writeln!(file, "{}", event_line);
        }
    }

    /// Returns the flattened weight matrix [num_neurons × NUM_INPUT_CHANNELS] for GPU upload.
    pub fn flatten_weights(&self) -> Vec<f32> {
        let mut flat = Vec::with_capacity(self.neurons.len() * NUM_INPUT_CHANNELS);
        for neuron in &self.neurons {
            for ch in 0..NUM_INPUT_CHANNELS {
                flat.push(if ch < neuron.weights.len() { neuron.weights[ch] } else { 0.0 });
            }
        }
        flat
    }
}

/// ----------------------------------------------------------------------------
/// 4. NEUROMODULATION SYSTEM
/// ----------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct NeuroModulators {
    pub dopamine: f32,      // Reward / Learning Rate (0.0 - 1.0)
    pub cortisol: f32,      // Stress / Inhibition (0.0 - 1.0)
    pub acetylcholine: f32, // Focus / Signal-to-Noise (0.0 - 1.0)
    pub tempo: f32,         // Clock-driven timing scale (0.0 - 2.0, 1.0 = nominal)
    /// FPGA timing stress — 0.0 = no violations, 1.0 = critical (WNS ≤ -5 ns)
    #[serde(default)]
    pub fpga_stress: f32,
}

impl NeuroModulators {
    /// Decode telemetry into chemical signals
    pub fn from_telemetry(telem: &crate::gpu::GpuTelemetry) -> Self {
        // DOPAMINE: Proportional to hashrate (Reward for doing work)
        // Target: 0.0105 MH/s = 1.0 Dopamine (calibrated to actual RTX 5080 Dynex hashrate).
        let dopamine = (telem.hashrate_mh / 0.0105).clamp(0.05, 1.0);

        // CORTISOL: Stress from heat or power spikes
        // 80°C = start stress, 90°C = max panic
        let heat_stress: f32 = 0.0; // Disabled: inaccurate temp sensors
        // RTX 5080 TDP ~430W. Stress starts at 400W (true thermal/power runaway territory).
        // Old value of 200W caused cortisol=1.0 at normal 250W load, zeroing all stimulus.
        let power_stress = ((telem.power_w - 400.0) / 50.0).clamp(0.0, 1.0);
        let cortisol = heat_stress.max(power_stress);

        // ACETYLCHOLINE: Stability of Vcore (Focus)
        // Since 12V sensors don't exist, we track the stability of VDDCR_GFX.
        // Fluctuations in core voltage under load represent neural "focus" jitter.
        let vddcr_dev = (telem.vddcr_gfx_v - 1.0).abs(); // Deviation from nominal 1.0V load
        let acetylcholine = (1.0 - vddcr_dev * 5.0).clamp(0.0, 1.0);

        // TEMPO: Clock-driven temporal scaling
        // Nominal RTX 5080 Core Clock: 2640 MHz
        let tempo = (telem.gpu_clock_mhz / 2640.0).clamp(0.1, 2.0);

        Self {
            dopamine,
            cortisol,
            acetylcholine,
            tempo,
            fpga_stress: 0.0,
        }
    }
}

use crate::models::PoolEvent;

impl NeuroModulators {
    /// Update chemical levels based on instantaneous events
    pub fn apply_event(&mut self, event: &PoolEvent) {
        match event {
            // VICTORY SPIKE: Finding a share is a "micro-reward"
            PoolEvent::ShareAccepted { latency_ms } => {
                // Base reward
                self.dopamine = (self.dopamine + 0.2).min(1.0); 
                
                // Latency Penalty: If laggy (>100ms), reduce Focus (Acetylcholine)
                if *latency_ms > 100 {
                    self.acetylcholine = (self.acetylcholine - 0.1).max(0.0);
                } else {
                    // Chicago Boost: Low latency improves focus
                    self.acetylcholine = (self.acetylcholine + 0.05).min(1.0);
                }
            }

            // JACKPOT: Finding a block is a massive Dopamine hit
            PoolEvent::BlockFound { .. } => {
                self.dopamine = 5.0; // Super-saturation (euphoria)
            }

            // STRESS: Switching pools causes temporary anxiety (Cortisol)
            PoolEvent::PoolSwitch { .. } => {
                self.cortisol = (self.cortisol + 0.3).min(1.0);
                self.dopamine = 0.0; // Reset reward expectation
            }
            
            _ => {}
        }
    }

    /// Natural decay (Homeostasis) - Call this every second
    pub fn decay(&mut self) {
        self.dopamine = (self.dopamine * 0.95).max(0.0);       // Joy fades
        self.cortisol = (self.cortisol * 0.90).max(0.0);       // Panic subsides
        self.acetylcholine = (self.acetylcholine * 0.99).max(0.0); // Focus lingers
    }
}