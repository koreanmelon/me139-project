# Reaction Wheel-Based Stability of a Three-Link Inverted Pendulum <!-- omit in toc -->

ME 139 Group 9: Connor Hennig, Jenny Mei, Kevin Cheng, Taewon Kim

## Table of Contents <!-- omit in toc -->

- [Project Description](#project-description)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

## Project Description

The main idea of this project is to design and create a bipedal robot that is able to raise and lower itself while remaining balanced.

## Quick Start

View the `src/case-1.ipynb` file to see the code for the first case study.

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
