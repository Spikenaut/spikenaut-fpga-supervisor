# Spikenaut FPGA Supervisor

A live hardware supervisor for Spikenaut that orchestrates GPU telemetry, FPGA brain pulses, and emergency brakes.

## Overview

The Spikenaut FPGA Supervisor is a Rust-based hardware orchestration system that provides real-time monitoring and control of neuromorphic hardware components. It manages GPU telemetry collection, FPGA communication, and implements spiking neural network (SNN) inference engines for advanced neuromorphic research.

## Features

- **GPU Telemetry**: Real-time monitoring of GPU performance metrics via NVML
- **FPGA Bridge**: Serial communication with FPGA hardware for brain pulse orchestration
- **Spiking Neural Networks**: Implementation of neuromorphic inference engines
- **Hardware Supervision**: Live monitoring and emergency brake systems
- **Research Tools**: Advanced neuromorphic research capabilities
- **Metrics Collection**: Prometheus-compatible metrics export
- **Process Safety**: Instance protection via lockfile mechanism

## Installation

### Prerequisites

- Rust 2024 edition
- CUDA-compatible GPU with NVML support
- FPGA hardware connected via serial port
- Linux operating system (tested on Linux)

### Build

```bash
cargo build --release
```

### Run

```bash
cargo run --bin spikenaut-supervisor
```

## Usage

The supervisor automatically detects FPGA hardware on standard USB serial ports (`/dev/ttyUSB0`, `/dev/ttyUSB1`, `/dev/ttyUSB2`) and initializes the monitoring systems.

### Command Line Options

The application supports environment-based configuration via the `clap` crate. Specific options can be viewed with:

```bash
cargo run --bin spikenaut-supervisor -- --help
```

## Architecture

### Core Modules

- **`gpu`**: Hardware bridge for GPU telemetry collection
- **`fpga`**: Serial communication bridge for FPGA devices
- **`snn`**: Spiking neural network inference engine
- **`research`**: Neuromorphic research tools and utilities
- **`trainer`**: Training utilities for neural networks
- **`cpu`**: CPU monitoring and metrics collection
- **`models`**: Data models for hardware and neural network components

### Key Components

1. **Hardware Bridge**: Abstract interface for GPU and FPGA communication
2. **Telemetry System**: Real-time metrics collection and export
3. **Inference Engine**: Optimized SNN implementation for neuromorphic computing
4. **Emergency Brakes**: Safety mechanisms for hardware protection

## Dependencies

### Core Dependencies
- `tokio`: Async runtime with full features
- `serde`: Serialization framework with derive support
- `tracing`: Structured logging and telemetry
- `metrics`: Metrics collection with Prometheus export
- `anyhow`: Error handling

### Hardware Interfaces
- `nvml-wrapper`: GPU monitoring via NVIDIA Management Library
- `serialport`: Serial communication for FPGA devices
- `nix`: System interfaces for signal handling

### External Libraries
- `spikenaut-fpga`: Custom FPGA library (local dependency)

## Configuration

The supervisor uses environment-based configuration. Key configuration areas include:

- Serial port paths for FPGA communication
- GPU monitoring parameters
- Metrics export endpoints
- Logging levels and outputs

## Monitoring

### Prometheus Metrics

The supervisor exports metrics compatible with Prometheus monitoring:

- GPU utilization and temperature
- FPGA communication status
- Neural network inference metrics
- System resource usage

### Logging

Structured logging via `tracing` with configurable output levels.

## Safety Features

- **Instance Protection**: Lockfile mechanism prevents multiple supervisor instances
- **Emergency Brakes**: Hardware protection mechanisms
- **Graceful Shutdown**: Proper resource cleanup on termination

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please ensure all submissions follow the project's coding standards and include appropriate tests.

## Hardware Requirements

- NVIDIA GPU with CUDA support
- FPGA device with serial communication capability
- Sufficient CPU resources for real-time processing

## Troubleshooting

### Common Issues

1. **Port Detection**: Ensure FPGA device is connected and accessible via `/dev/ttyUSB*`
2. **GPU Access**: Verify NVML installation and proper permissions
3. **Instance Conflicts**: Check for existing supervisor processes using lockfile

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
RUST_LOG=debug cargo run --bin spikenaut-supervisor
```

## Research Applications

This supervisor is designed for neuromorphic research applications including:

- Spiking neural network experimentation
- Hardware-software co-design
- Real-time neuromorphic inference
- Brain-inspired computing research
