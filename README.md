# Reaction Wheel-Based Stability of a Three-Link Inverted Pendulum <!-- omit in toc -->

ME 139 Group 9: Connor Hennig, Jenny Mei, Kevin Cheng, Taewon Kim

## Table of Contents <!-- omit in toc -->

- [Project Description](#project-description)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

## Project Description

The main idea of this project is to design and create a bipedal robot that is able to raise and lower itself while remaining balanced.

## Quick Start

Make sure you have Python >3.10.6 installed. Create a virtual environment with
```bash
$ python3 -m venv .venv
```

Activate the virtual environment with
```bash
$ source .venv/bin/activate
```

After activating the virtual environment, install the required packages with
```bash
$ pip3 install -r requirements.txt
```

To simulate any of the existing systems with default settings, run the following command in the root directory.
```bash
$ python3 main.py ip1lrw --q0 0 0 0 0
```

An example command with some additional settings is shown below.
```bash
$ python3 main.py ip1lrw --q0 0 0 0 0 --fps 60 --duration 30 --save --no-show
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

- `animator/` handles rendering and saving the simulation
- `simulator/` solves an IVP for a given system
- `systems/` contains the code that defines the system dynamics for various robots
