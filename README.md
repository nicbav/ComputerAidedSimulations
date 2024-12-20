# ComputerAidedSimulations
There are some simulations carried out by me in the Computer Aided Simulations course, e.g. Prey/Predator Model, SuperMarket Model
Supermarket Simulation
This simulation models a supermarket's customer flow and queue management system. It uses a discrete event simulation approach to analyze customer interactions across various departments, including Fresh Food, Fresh Meat, and Cash Registers. The primary objective is to study queue sizes, service times, and delays.

Features
Event-driven Model: Simulates arrivals, service, and departures of customers.
Departments Simulated: Fresh Food, Fresh Meat, Cash Registers.
Performance Metrics: Total delays, average queue sizes, and confidence intervals for multiple runs.
Randomized Paths: Customers follow different shopping routes through departments based on probabilities.
How It Works
Event Scheduling: Events (e.g., arrivals, departures) are managed using a priority queue.
Random Variables: Customer arrival and service times are modeled with exponential distributions.
Queue Management: Tracks queue sizes and delays at each department.
Statistical Analysis: Aggregates metrics over multiple runs to calculate averages and confidence intervals.
Configuration
Input Parameters: Number of servers per department, arrival rate, and service rates.
Run Parameters: Simulation duration, number of runs, and confidence level for statistical analysis.
Output
Mean and confidence intervals for:
Total delay.
Queue sizes at each department.
Delays experienced by customers.
Histograms and detailed logs for optional visualization.
This simulation can be extended or tailored for specific use cases in retail or service optimization scenarios.


