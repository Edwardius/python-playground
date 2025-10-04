# Motor Response Analysis Scripts

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Add control input to data
```bash
python add_control_input.py 0_to_20.txt
```
Creates `0_to_20_with_control.txt` with step function control input.

### 2. Plot data
```bash
python plot_angle_data.py 0_to_20_with_control.txt
```
Interactive plot with zoom and hover tooltips.

### 3. Analyze motor response
```bash
python analyze_motor_response.py filename start_ms end_ms Kp [--debug]
```

Example:
```bash
python analyze_motor_response.py 0_to_20_with_control.txt 2000 6000 1.5
python analyze_motor_response.py 0_to_20_with_control.txt 2000 6000 1.5 --debug
```

- `filename`: Data file with control input
- `start_ms`, `end_ms`: Time range to analyze
- `Kp`: Proportional gain of controller  
- `--debug`: Show plot with processing stages and system parameters
