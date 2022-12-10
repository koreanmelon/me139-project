# Reaction Wheel-Based Stability of a Three-Link Inverted Pendulum <!-- omit in toc -->

ME 139 Group 9: Connor Hennig, Jenny Mei, Kevin Cheng, Taewon Kim

## Table of Contents <!-- omit in toc -->

- [Project Description](#project-description)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

## Project Description

The main idea of this project is to design and create a bipedal robot that is able to raise and lower itself while remaining balanced.

## Quick Start

To simulate any of the existing systems with default settings, run the following command in the root directory.
```bash
$ python3 main.py reactionwheel 
```

An example command with some additional settings is shown below.
```bash
$ python3 main.py reactionwheel --fps 60 --duration 30 --save --no-show
```

For help with any parameters, simply run
```bash
$ python3 main.py --help
```

## Project Structure

The project is organized as follows:

- `assets/` contains some diagrams and simulation outputs
- `cache/` contains some cached data to speed up computation/simulation time
- `docs/` contains the documentation for the project (NOT CREATED YET)
- `notebooks/` contains relevant Jupyter notebooks
- `src/` contains the source code
- `tests/` contains the unit tests

The `src/` directory is organized as follows:

- `simulator/` contains the simulator code which solves an IVP for a given system
- `animation/` contains the code that renders and saves the simulation
- `systems/` contains the code that defines the system dynamics for various robots
