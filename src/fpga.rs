use serialport;
use std::time::Duration;
use std::io::{Read, Write};

/// FPGA Bridge — Communicates with the Basys 3 SNN via UART
pub struct FpgaBridge {
    port: Box<dyn serialport::SerialPort>,
}

impl FpgaBridge {
    /// Initialize the serial port at 115200 Baud
    pub fn new(port_name: &str) -> Result<Self, anyhow::Error> {
        let port = serialport::new(port_name, 115200)
            .timeout(Duration::from_millis(1000))
            .open()?;
            
        Ok(Self { port })
    }

    /// Perform a SiliconBridge v3.0 Neural Cluster Step (16 Neurons):
    /// 1. Send Sync Byte (0xAA)
    /// 2. Send 16x 16-bit Q8.8 stimuli (32 bytes)
    /// 3. Receive 36-byte frame containing:
    ///    - [1 Sync] + [32 Potentials] + [2 Spike Flags] + [1 Checksum]
    pub fn step_cluster(&mut self, stimuli: &[u16; crate::NUM_NEURONS]) -> Result<(Vec<u16>, u16), anyhow::Error> {
        let mut tx_data = Vec::with_capacity(33);
        tx_data.push(0xAA); // Resilient Protocol v2.0: Sync Header
        for &s in stimuli {
            tx_data.push((s >> 8) as u8);
            tx_data.push(s as u8);
        }
        let mut rx_data = [0u8; 36];
        self.port.read_exact(&mut rx_data)?;
        
        // 4. Validate Checksum (XOR Sum)
        let mut calculated_checksum = 0u8;
        for i in 0..35 {
            calculated_checksum ^= rx_data[i];
        }
        
        if calculated_checksum != rx_data[35] {
            return Err(anyhow::anyhow!("SiliconBridge v3.0: Checksum Mismatch! (Expected {:02X}, got {:02X})", 
                calculated_checksum, rx_data[35]));
        }

        let mut potentials = Vec::with_capacity(crate::NUM_NEURONS);
        for i in 0..crate::NUM_NEURONS {
            let p = ((rx_data[i*2 + 1] as u16) << 8) | (rx_data[i*2 + 2] as u16);
            potentials.push(p);
        }
        let spikes = ((rx_data[33] as u16) << 8) | (rx_data[34] as u16);

        Ok((potentials, spikes))
    }
}
