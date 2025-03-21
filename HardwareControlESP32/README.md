# How to Run the Robot Simulation

<!-- > **Note:** This setup is still in progress and might require adjustments. Please ensure that each step works as expected before proceeding. -->

## Table of Contents

- [Prerequisites](#prerequisites)
- [Software Installation](#software-installation)
- [Running the Simulation](#running-the-simulation)
- [Output & Results](#output--results)

---

## Prerequisites

Before you start, ensure you have the following dependencies installed:

- **Python 3.x**  
  Ensure you are using Python 3.6 or higher to avoid compatibility issues.
- **System Requirements**  
  Make sure your system has sufficient memory and processing power to run the simulation. 
---

## Software Installation

To install the required libraries, run the following command:

```bash
pip install scipy matplotlib mujoco imageio[ffmpeg] # imageio\[ffmpeg\] in zsh 
```
## Running the Simulation

Once the libraries are installed, navigate to the `code` directory and execute each of the following Python scripts in sequence:

1. **Navigate to the `code` directory:**

```bash
cd code
```

```bash
python 1_snake_forward_ft_final.py
python 1_snake_rotate_ft.py
python 2_biped_rotate_ft.py
python 3_dual_snake_rotate_ft.py
python 4_dual_snake_forward_ft.py
python 5_dual_snake_worm_ft.py
python 6_forward_biped_sucess.py
```

## Output & Results

- **Videos:** Each script will generate a corresponding video of the simulation in the same directory.
- **Auxiliary Data:** Additional data related to the robot's movements, configuration, and other metrics will also be saved in the same directory.

