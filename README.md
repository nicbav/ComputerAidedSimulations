# ComputerAidedSimulations
There are some simulations carried out by me in the Computer Aided Simulations course, e.g. Prey/Predator Model, SuperMarket Model
# Supermarket Simulation

This project is a discrete event simulation model of a supermarket's customer flow and queue management system. It aims to analyze and optimize the operations of various departments, such as Fresh Food, Fresh Meat, and Cash Registers.

## Features

- **Event-driven Simulation**: Models customer arrivals, service, and departures using a priority queue.
- **Departments Included**:
  - Fresh Food.
  - Fresh Meat.
  - Cash Registers.
- **Metrics Collected**:
  - Total delays experienced by customers.
  - Average queue sizes for each department.
  - Confidence intervals for performance metrics across multiple simulation runs.
- **Randomized Customer Paths**: Customers follow probabilistic routes through different departments.

## How It Works

1. **Event Scheduling**: 
   - Events (e.g., arrival, departure) are stored and processed in a priority queue based on their timestamps.
   
2. **Random Variables**:
   - Arrival and service times are generated using exponential distributions to simulate realistic variability.

3. **Queue Management**:
   - Tracks the number of customers in line and delays at Fresh Food, Fresh Meat, and Cash Registers.

4. **Statistical Analysis**:
   - Runs multiple simulations to compute averages and confidence intervals for performance metrics.

## Configuration

### Input Parameters
- Number of servers for:
  - Fresh Food (`NUM_SERVERS_FOOD`).
  - Fresh Meat (`NUM_SERVERS_MEAT`).
  - Cash Registers (`NUM_SERVERS_CASH`).
- Arrival rate (`ARRIVAL`) and service rates for each department.

### Run Parameters
- Simulation time (`max_time`).
- Number of simulation runs (`NUM_RUNS`).
- Confidence level for statistical analysis (`CONF_LEVEL`).

## Output

The simulation generates:
- Mean and confidence intervals for:
  - Total delay.
  - Average queue sizes at each department.
  - Delays experienced by customers at each stage.
- Optional histograms for visualizing delays and queue sizes.

## Usage

1. Clone the repository.
2. Configure the input parameters in the script.
3. Run the simulation:
   ```bash
   python SuperMarketModel.py


