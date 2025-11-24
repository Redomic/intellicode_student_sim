# Student Simulation (Simulated Learners)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Nano-GraphRAG](https://img.shields.io/badge/GraphRAG-Nano-blue?style=for-the-badge)

> **Acknowledgment**: This module is an adaptation of **"Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents"** (ACL 2025) by Wu et al. We have integrated their cognitive simulation architecture into IntelliCode to validate our adaptive tutoring system with diverse simulated learner profiles.

---

## Overview

This module provides the **Simulated Learner** agents used to evaluate the IntelliCode platform. By generating agents with specific cognitive profiles (mastery levels, memory retention rates, learning velocities), we can run longitudinal simulations to test the efficacy of our adaptive curriculum and feedback mechanisms without requiring human subjects for initial validation.

## Key Components

*   **Cognitive Behavior Modeling**: Simulates learner behavior including misconceptions, forgetting curves, and fatigue.
*   **IntelliT Adapter**: Custom bridge connecting the simulation engine to the IntelliCode backend API.
*   **Metrics Tracking**: Logs learning trajectories, hint usage, and mastery gains for analysis.

## Setup & Usage

### Prerequisites

*   Python 3.10+
*   IntelliCode Backend running (usually on `http://localhost:8000`)

### Installation

1.  Navigate to the simulation directory:
    ```bash
    cd student_sim
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running Simulations

To run a simulation with the IntelliCode integration:

```bash
# Run the main simulation script
python main_intellit.py
```

This script will:
1.  Initialize a simulated student with a specific profile.
2.  Connect to the local IntelliCode backend.
3.  Simulate a multi-session learning trajectory (solving problems, requesting hints, taking breaks).
4.  Log results to the `results/` directory.

## Configuration

Configuration is managed in `config_intellit.py`:

*   **`API_BASE_URL`**: URL of the IntelliCode backend.
*   **`SIMULATION_DAYS`**: Number of simulated days to run.
*   **`LEARNER_PROFILES`**: Definitions for different learner archetypes (e.g., "Fast Learner", "Struggling Novice").

## Original Citation

If you use this simulation framework, please cite the original work:

```bibtex
@inproceedings{studentsim,
  author       = {Tao Wu and Jingyuan Chen and Wang Lin and Mengze Li and Yumeng Zhu and Ang Li and Kun Kuang and Fei Wu},
  title        = {Embracing Imperfection: Simulating Students with Diverse Cognitive Levels Using LLM-based Agents},
  booktitle    = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year         = {2025}
}
```
