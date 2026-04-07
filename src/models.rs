use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FaultClass {
    MlccShortLoad,
    VrmPhaseShort,
    FirmwareStateBad,
    GpuDieDead,
    HealthyStandby,
    UnknownAmbiguity,
}

impl FaultClass {
    pub fn to_code(&self) -> &str {
        match self {
            Self::MlccShortLoad => "ERR_MLCC_001",
            Self::VrmPhaseShort => "ERR_VRM_PHASE",
            Self::FirmwareStateBad => "ERR_FW_CRC_FAIL",
            Self::GpuDieDead => "ERR_SILICON_DEAD",
            Self::HealthyStandby => "STATUS_OK",
            Self::UnknownAmbiguity => "WARN_AMBIGUOUS",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolEvent {
    ShareAccepted { latency_ms: u32 },
    BlockFound { height: Option<u64> },
    JobUpdate { difficulty: u64 },
    PoolSwitch { url: String },
    ShareRejected { reason: String },
    None,
}

// ── Mining Fault Detection ────────────────────────────────────────────

/// Mining-specific fault classification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MiningFaultClass {
    HealthyMining,
    ThermalThrottling,
    StaleSolutions,
    CudaException,
    NetworkLag,
}

/// Solver statistics for mining operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverStats {
    pub steps_per_second: f64,
    pub solutions_found: u64,
    pub current_step: u64,
    pub nonce: u64,
}

/// Mining diagnostic report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningDiagnostic {
    pub fault: MiningFaultClass,
    pub timestamp: std::time::SystemTime,
}

/// Mining monitor for tracking solver performance
pub struct MiningMonitor {
    pub device_id: u32,
    pub last_stats: Option<SolverStats>,
}

impl MiningMonitor {
    pub fn new(device_id: u32) -> Self {
        Self {
            device_id,
            last_stats: None,
        }
    }

    pub fn reset(&mut self) {
        self.last_stats = None;
    }

    pub fn sample(&mut self, stats: Option<&SolverStats>) -> MiningDiagnostic {
        if let Some(s) = stats {
            self.last_stats = Some(s.clone());
        }

        // Mock diagnostic for now - assume healthy
        MiningDiagnostic {
            fault: MiningFaultClass::HealthyMining,
            timestamp: std::time::SystemTime::now(),
        }
    }

    pub fn save_diagnostics(&self, _path: &str) -> std::io::Result<()> {
        // Todo: Serialize history to JSON
        Ok(())
    }
}
