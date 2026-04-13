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
git clone https://github.com/Langyuan-Wu/HydroSpecFit.git
cd HydroSpecFit
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

HydroSpecFit requires the following packages:

* customtkinter
* pandas
* numpy
* scipy
* matplotlib
* openpyxl

See `requirements.txt` for the full dependency list.

---

## Usage

Run the main program:

```bash
python HydroSpecFit.py
```

The graphical user interface will open.

---

## Sample Data

One complete example dataset group is provided in the [`sample_data`](./sample_data) folder:

- [`qcmd_example.xlsx`](./sample_data/qcmd_example.xlsx): QCM-D data
- [`echem_example.xlsx`](./sample_data/echem_example.xlsx): electrochemical data
- [`quartz_in_air_example.xlsx`](./sample_data/quartz_in_air_example.xlsx): quartz reference signal measured in air

These three files belong to the same experiment and should be used together to test the full HydroSpecFit workflow.

Detailed information for this example dataset, including the experimental context and recommended parameter values, is provided in [`sample_data/README.md`](./sample_data/README.md).

---

## Input Files

HydroSpecFit accepts Excel files (`.xlsx`) as input.

### 1. QCM-D data

Required columns include:

| Type        | Accepted examples                                                           |
| ----------- | --------------------------------------------------------------------------- |
| Time        | `time`, `Time`, `t`, `sec`, `seconds`, `Time [sec]`, `AbsTime`, `RelTime`   |
| Frequency   | `f3`, `f5`, `f7`, `f9`, `f11`, ...                                          |
| Dissipation | `D3`, `D5`, `D7`, `D9`, `D11`, ... or lowercase versions such as `d3`, `d5` |

### 2. Electrochemical data

Required columns include:

| Type      | Accepted examples                                 |
| --------- | ------------------------------------------------- |
| Time      | `time`, `Time`, `t`, `sec`, `seconds`, `time/s`   |
| Potential | `Ewe`, `Potential`, `Voltage`, `<Ewe/V>`, `Ewe/V` |
| Charge    | `Q`, `Charge`, `C`, `(Q-Qo)`, `(Q-Qo)/mC`, `Q/mC` |

### 3. Reference quartz files

The software also supports reference quartz files containing stabilized frequency and dissipation signals, for example:

* Quartz in air

Reference files should contain columns such as:

```text
f3, f5, f7, f9, f11
D3, D5, D7, D9, D11
```

The last valid value in each column is used as the stabilized reference signal.

---

## Recommended Workflow

1. Load QCM-D data
2. Load electrochemical data
3. Load quartz reference data
4. Check physical, electrochemical, and electrolyte parameters
5. Use **Time Sync** if QCM-D and EChem data need alignment
6. Open **Auto-Cycles Optimization**
7. Define cycle-wise or segment-wise viscosity and coverage (`θ`)
8. Run dynamic optimization
9. Inspect combined plots of `Δf/n`, `ΔD`, `h`, and `ξ`
10. Double-click a point to inspect row-level fitting details
11. Export figures and Excel results

---

## Main Interface Modules

### Time Sync

Allows manual or automatic alignment of QCM-D and electrochemical time axes.

### Global / Cycle Viscosity Calibration

Uses a Kanazawa-based fitting routine to estimate liquid viscosity from selected baseline points.

### Auto-Cycles Optimization

Detects cycles automatically and allows the user to assign viscosity and coverage values for each cycle or segment.

### Combined Analysis Window

Displays:

* `Δf_n / n`
* `ΔD_n`
* optimized `h`
* optimized `ξ`

### Row Analysis Window

Displays model-vs-experiment fitting details for a selected row.

---

## Outputs

HydroSpecFit can export:

* Combined analysis images (`PNG` / `PDF`)
* Row analysis images (`PNG` / `PDF`)
* Final fitting results (`Excel`)
* Row-level model/experiment comparison data (`Excel`)

Typical exported result columns include:

* optimized height (`h`)
* optimized permeability length (`ξ`)
* cutoff value
* fit status
* used viscosity
* used coverage (`θ`)

---

Suggested contents:

* `assets/`: screenshots of the GUI and analysis windows
* `sample_data/`: one complete example dataset group for learning and testing
* `example_outputs/`: representative exported figure and data results

---

## Notes

* Input files must be provided in Excel format.
* Column names should follow the accepted naming patterns listed above.
* Electrochemical data are required for theoretical curve calculation and cycle detection.
* Reference frequency and dissipation values must be provided for all active harmonics.
* For complex non-Newtonian systems, manual segmentation may be required.

---

## Authors

* Langyuan Wu
* Avishay Bukhman

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](./LICENSE) file for details.


