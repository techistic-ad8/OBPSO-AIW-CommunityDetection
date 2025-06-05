# OBPSO-AIW: Enhanced PSO for Community Detection  
*A smarter approach to uncovering hidden groups in social networks*

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![Jupyter](https://img.shields.io/badge/Built%20With-Jupyter-orange)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

---

## 🔍 About the Project

This project is about finding **communities** in social networks — like discovering your school friend group, your gaming buddies, or family circles in a huge crowd.

To do this, we use a smart optimization algorithm called **Particle Swarm Optimization (PSO)** — but better. We created **OBPSO-AIW**, an enhanced version that finds communities more accurately and faster.

We added 3 key features:
- ✨ **Opposition-Based Learning (OBL)** — For smarter starting points
- 🎯 **Adaptive Inertia Weight (AIW)** — To balance exploring vs improving
- 🔧 **Local Search** — To fine-tune results to be the best

---

## 💡 What is Community Detection?

In a network:
- **People = Nodes**
- **Interactions = Edges**

A **community** is a group of people more connected to each other than to outsiders.

Detecting communities helps with:
- Friend/content recommendations 📲
- Finding influencers 💬
- Spotting fake accounts 🔐
- Better targeted marketing 🎯

---

## ⚙️ Why We Enhanced PSO

Regular PSO is like birds searching for food — they follow each other, hoping someone finds it.

But it has problems:
- Gets stuck on the first good solution 🐌
- Doesn’t explore enough 🕵️
- Becomes lazy too soon 😴

**OBPSO-AIW fixes that.**

---

## 🔧 How OBPSO-AIW Works

### Step-by-Step Flow:

1. **Initialization with OBL**  
   Start with both normal and opposite solutions → pick better ones

2. **Fitness Evaluation**  
   Measure how good each solution is using **modularity** (higher = better communities)

3. **Adaptive Velocity Update**  
   Particles move smartly using decreasing **inertia weight**

4. **Local Search**  
   Improve the best solutions by minor tweaks

5. **Track Bests**  
   Update personal best and global best results

6. **Repeat**  
   Until best community structure is found

---

## 🧪 Experiment Setup

| Detail              | Value                                         |
|---------------------|-----------------------------------------------|
| Language/Platform   | Python 3.12 (Jupyter Notebook)                |
| Libraries           | NumPy, NetworkX, Scikit-learn                 |
| System              | AMD Ryzen 7, 16GB RAM, NVIDIA RTX 3070 GPU    |
| Datasets Used       | Zachary’s Karate Club, Dolphin, Football, Strike |
| Evaluation Metrics  | Modularity, NMI, ARI, Execution Time, MNDN    |

---

## 🧠 Algorithm Summary

> “Think of birds flying in a sky, but smarter birds that learn from each other, explore new paths, and always try to find the best flock — that’s OBPSO-AIW.”

Key Concepts:
- **Modularity (Q)**: Measures community quality
- **Binary Encoding**: Each node assigned to a group
- **Crossover & Mutation**: Swapping solutions to explore better ones
- **Diversity First**: OBL gives more starting variety
- **Refinement Last**: Local tweaks finalize top solutions

---
## 📌 Key Parameters

| Parameter            | Value Range     | Description                                   |
|----------------------|------------------|-----------------------------------------------|
| Population Size       | 50 – 100         | Number of particles (possible solutions)      |
| Crossover Rate        | 0.7 – 0.9        | Fraction of individuals to cross over         |
| Mutation Rate         | 0.01 – 0.1       | Fraction to mutate (split/fuse)               |
| Inertia Weights (ω)   | 0.9 → 0.4        | Starts high (explore) → lowers (exploit)      |
| Max Iterations        | 100 – 500        | Controls number of search cycles              |

---

## 📈 Results Snapshot

Our algorithm outperformed:
- ✅ Standard PSO
- ✅ SGO (Simple Genetic)
- ✅ OBGA (Opposition-Based GA)
- ✅ SSA (Salp Swarm Algorithm)
- ✅ MCOBGA (Modified Genetic variant)

✔ Higher modularity  
✔ Faster convergence  
✔ Better-defined communities

---


## 📂 Project Structure

📦 OBPSO-AIW-CommunityDetection
├── 📂 src/ → Core algorithm files
│ ├── standard_pso.py → Baseline PSO implementation
│ ├── enhanced_pso.py → OBPSO-AIW logic
| ├── run_standard_pso.py → Run baseline PSO
| ├── run_obpso_aiw.py → Run our enhanced algorithm    
│ └── utils.py → Helper functions
├── 📂 data/ → Datasets
├── 📂 results/ → Output graphs and logs
├── requirements.txt → Python dependencies
└── README.md → You’re reading it!
