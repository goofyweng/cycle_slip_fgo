# GNSS Cycle Slip Detection and Identification

## Description
This project explores and implements techniques for detecting and identifying cycle slips in GNSS data. Cycle slips occur when there is a sudden change in the carrier-phase measurements due to signal interruptions, receiver dynamics, or atmospheric effects. 

To enhance the robustness of GNSS positioning, this project employs **Factor Graph Optimization (FGO)** to estimate the **receiver Position, Velocity, and Time (PVT)** solution. The optimization process integrates **pseudorange** and **Time Difference Carrier Phase (TDCP)** observations to improve accuracy and mitigate cycle slip effects.

## Features
- Implementation of cycle slip detection and identification techniques.
- Use of **Factor Graph Optimization (FGO)** for enhanced PVT estimation.
- Processing of **pseudorange** and **TDCP** observations.
- Simulation and analysis of GNSS cycle slip scenarios.

## Installation

### Prerequisites
- Python (>=3.11)
- Virtual environment support (built into Python >=3.11)

### Setup Instructions

1. **Clone the repository**:
   ```sh
   git clone https://github.com/goofyweng/cycle_slip_fgo.git
   cd cycle_slip_fgo
2. **Create and activate a virtual environment:**:
    ```sh
    python -m venv venv
- On Linux/macOS:
    ```sh
    source venv/bin/activate
- On Windows (cmd):
    ```sh
    venv\Scripts\activate
3. **Install dependencies**:
    ```
    pip install -r requirements.txt







