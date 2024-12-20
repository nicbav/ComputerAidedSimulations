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

# PreyPredator Simulation

This project implements a simulation model for studying population dynamics in a prey-predator ecosystem. The model uses stochastic events to represent reproduction, predation, and mortality, allowing analysis of population trends and interactions.

## Features

- **Event-driven Simulation**:
  - Prey (herbivores) and predators (carnivores) dynamically interact through events such as reproduction, predation, and natural death.
- **Population Tracking**:
  - Separate counts for male and female populations of both species.
- **Configurable Scenarios**:
  - Simulation runs for a defined number of events or until populations are extinct.
- **Statistical Analysis**:
  - Confidence intervals calculated over multiple runs for key metrics.

## How It Works

### Key Events
1. **Herbivorous Reproduction**:
   - Male and female herbivores reproduce probabilistically, adding offspring to the population.
2. **Carnivorous Reproduction**:
   - Male and female carnivores reproduce similarly, contributing to predator numbers.
3. **Predation**:
   - Carnivores randomly hunt herbivores, reducing their population.
4. **Natural Death**:
   - Both species experience mortality due to competition or natural causes.

### Simulation Modes
- **Mode A**: Single simulation with default initial populations.
- **Mode B**: Ten simulation runs, followed by confidence interval calculations for key metrics:
  - Reproduction events.
  - Predation events.
  - Death events for both species.

### Input Parameters
- **Population Thresholds**:
  - `herbivore_competition_threshold`: Maximum herbivore population before competition increases mortality.
  - `carnivore_competition_threshold`: Predator-to-prey ratio beyond which carnivores experience additional mortality.
- **Event Rates**:
  - Initial rates for reproduction, predation, and mortality events.

## Outputs

- **Mode A**:
  - Graph showing population dynamics over time for herbivores and carnivores.
- **Mode B**:
  - Confidence intervals for reproduction, predation, and death events.
  - Bar chart summarizing confidence intervals for key metrics.



