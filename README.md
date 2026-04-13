# HydroSpecFit

#### A graphical Python tool for hydrodynamic spectroscopy analysis of multiharmonic EQCM-D data.

HydroSpecFit is a GUI-based Python program for analyzing multiharmonic electrochemical quartz crystal microbalance with dissipation monitoring (EQCM-D) data. It integrates QCM-D signals, electrochemical data, and reference quartz signals to extract time-resolved hydrodynamic descriptors, including film thickness (`h`) and permeability length (`ξ`), through hydrodynamic model fitting.

The program provides an interactive workflow for data import, time synchronization, viscosity calibration, automatic cycle detection, cycle- or segment-specific parameter assignment, model fitting, and figure/data export.

---

## Features

- Import QCM-D data from Excel files
- Import electrochemical data from Excel files
- Import quartz reference signals measured in air/material/coated material states
- Automatic harmonic detection from QCM-D frequency columns
- Time synchronization between QCM-D and electrochemical datasets
- Kanazawa-based electrolyte viscosity calibration
- Automatic cycle detection from electrochemical potential data
- Dynamic cycle-wise and segment-wise viscosity/coverage settings
- Hydrodynamic fitting using a porous-layer model
- Extraction of optimized film thickness `h` and permeability length `ξ`
- Combined analysis plots for `Δf/n`, `ΔD`, `h`, and `ξ`
- Interactive row-by-row fitting inspection
- Export of figures as PNG/PDF
- Export of results as Excel files

---

## Installation

Clone the repository:

```bash
git clone https://github.com/langyuanwu/HydroSpecFit.git
cd HydroSpecFit
