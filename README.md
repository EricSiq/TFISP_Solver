# TFISP Solver

A command-line tool for solving the **Tactical Fixed Interval Scheduling Problem (TFISP)** using both classical and quantum/quantum-inspired optimization methods.

---

## Overview

TFISP Solver transforms scheduling problems — where tasks have fixed times and limited resources — into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.  
It supports solvers ranging from **classical optimizers** like CPLEX and Gurobi to **quantum/hybrid** solvers like D-Wave.

---

## Real-World Applications

- **Manufacturing** – Assign tasks to machines without overlaps, minimizing activation costs.  
- **Healthcare** – Schedule surgeries in limited operating rooms without conflicts.  
- **Data Centers** – Allocate jobs to servers efficiently.  
- **Logistics** – Plan deliveries or maintenance schedules.  
- **Events** – Organize sessions into available time slots without clashes.  

---

## How It Works

1. **Problem Input** – Define jobs, resources, and constraints in a text file:
   ```txt
   NUM_JOBS=5
   NUM_RESOURCES=3
   RESOURCE_COSTS=100.0,200.0,150.0

   JOBS:
   0, 10, 20
   1, 15, 25
   2, 30, 40
   3, 35, 45
   4, 5, 15

   DISQUALIFIED_RESOURCES:
   0, 2
   1, 0
   1, 1
   2, 0
   3, 1
