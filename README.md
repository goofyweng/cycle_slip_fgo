# GNSS Cycle Slip Detection and Identification

## Description
This project explores and implements techniques for detecting and identifying cycle slips in GNSS data. Cycle slips occur when there is a sudden change in the carrier-phase measurements due to signal interruptions, receiver dynamics, or atmospheric effects. 

To enhance the robustness of GNSS positioning, this project employs **Factor Graph Optimization (FGO)** to estimate the **receiver Position, Velocity, and Time (PVT)** solution. The optimization process integrates **pseudorange** and **Time Difference Carrier Phase (TDCP)** observations to improve accuracy.

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
    ```
## Usage
1. **Toy examples**
Run `toy_examples.py` `toy_examples1_aichi.py` `toy_examples2_aichi.py` `toy_examples3_aichi.py` `toy_examples4_aichi.py`to show the toy examples for single fault detection and single fault identification. The results are shown as confusion matrix.
    ```
    python toy_examples.py
    ```
2. **GNSS example**
- Run `GNSS_example.py` to calculate weighted residual norm \(z\) and detection threshold \(T\) and show on a central chi-squared distribution.
    ```
    python GNSS_example.py
    ```
- Run `GNSS_example_z_hist_sim_data.py` to show the histogram of calculated weighted residual norm \(z\) using simulated pseudorange and TDCP observation. Be aware of the long processing time for this python script.
    ```
    python GNSS_example_z_hist_sim_data.py
    ```
- Run `GNSS_example_build_table_find_domi_sat.py` to find the geometrically dominate satellite by manually adding a single fault to different satellite PRN.
    ```
    python GNSS_example_z_hist_sim_data.py
    ```
- Run `GNSS_example LLI1.py` to show the single fault detection result in a confusion matrix using real-world GNSS data.
    ```
    python GNSS_example LLI1.py
    ```

