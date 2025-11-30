# Hot/Cold B-tree Index with Machine Learning–Based Adaptive Sampling  
CS 543 — Project 2 (Fall 2025)

This repository contains a standalone implementation of a **Hot/Cold B-tree Index (HCIndex)**, extended with **machine-learning–driven adaptive sampling** for dynamic promotion of hot keys into a fast in-memory hot tier. The project is built around the idea that real-world workloads are often highly skewed, meaning a small subset of keys receives a large portion of queries. By learning which keys deserve fast-path treatment, the system can reduce logical work per query and improve performance.

---

## Overview

The system maintains two B-trees:

1. **Cold B-tree**  
   Contains *all* keys and acts as the fallback index.

2. **Hot B-tree**  
   Contains frequently accessed keys that have been promoted based on hit scores and a sampling policy.

Hits, node visits, and promotions are tracked for analysis.

A key component of the system is the **sampling rate (D)**, which determines the probability of promoting a key from the cold tree into the hot tree once it becomes sufficiently “hot”.

---

## Machine Learning Approaches for Adaptive Sampling

The project explores **three different adaptation strategies** for adjusting the sampling rate \(D\).  
These represent increasing levels of sophistication, moving from simple heuristics to true ML and RL-based learning.

### 1. Heuristic Hill-Climbing (Initial Approach)

The first approach used a simple rule-based hill-climbing strategy.  
After every fixed number of queries, the system measured the change in cost (node visits per query) relative to the previous interval. If cost decreased when the sampling rate \(D\) was increased previously, the system increased \(D\) again. If cost increased, the system reversed direction.  

This method did not learn a predictive model and relied only on short-term fluctuations. As a result, it was unstable under noise and often adjusted the sampling rate in unhelpful ways.

---

### 2. Online Linear Regression with Stochastic Gradient Descent (Second Approach)

The second approach implemented a classical ML method: **online linear regression**.  
The system attempted to learn a cost function of the form:

\[
\widehat{\text{cost}}(D) = w_0 + w_1 D
\]

The weights \(w_0\) and \(w_1\) were updated using stochastic gradient descent based on observed prediction errors.  
After each interval, the model used the sign of the learned slope \(w_1\) to decide whether to increase or decrease \(D\).

Although this was a true ML model, the cost surface turned out to be noisy and poorly modeled by a linear function, causing the system to learn an incorrect relationship and push \(D\) toward very small values.

---

### 3. Epsilon-Greedy Multi-Armed Bandit (Final and Improved Approach)

The third and most successful approach uses an **epsilon-greedy multi-armed bandit**, a simple reinforcement-learning technique.  
Instead of modeling cost as a continuous function of \(D\), the system evaluates a small set of discrete sampling-rate options (e.g., \(D = 0.3, 0.5, 0.7, 1.0\)). Each value of \(D\) is treated as an “arm” of the bandit.  

For each arm, the system keeps track of the average cold-node cost observed when that sampling rate is used. Using an ε-greedy strategy, the controller occasionally explores different sampling rates but otherwise exploits the one with the lowest observed cost.

This approach avoids incorrect assumptions about the cost function, is stable under noise, and learns a good sampling rate for the workload.

---

## Repository Structure

