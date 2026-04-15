import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import os
import time
import re

# App Appearance Settings
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# Disable standard Matplotlib Toolbar
plt.rcParams['toolbar'] = 'None' 

# ==========================================
# --- 1. Physics Engine ---
# ==========================================

def q0_cal(dn):
    return (1+1j) / dn

def q1_cal(q0, qsi):
    qsi = np.maximum(qsi, 1e-12) 
    return np.sqrt(q0**2 + (1/(qsi**2)))

def dalta_model(dn, thata, qsi_val, h_val, Liquid_Density, Quartz_Density, Quartz_Viscosity):
    try:
        q0 = q0_cal(dn)
        q1 = q1_cal(q0, qsi_val)
 
        arg = q1 * h_val
        if np.max(np.real(arg)) > 100: return None, None

        ch = np.cosh(arg)
        sh = np.sinh(arg)   
        A_val = q1 * ch + q0 * sh
        
        if np.any(np.abs(A_val) < 1e-15): return None, None

        part1 = 1/q0
        part2 = h_val / ((qsi_val * q1)**2)
        part3a = 1 / (A_val * ((qsi_val * q1)**2))
        part3b = ((2 * q0) / q1) * (ch - 1) + sh
        part3 = part3a * part3b
        val_total = part1 + part2 - part3
        
        coeff = 1 / np.sqrt(Quartz_Viscosity * Quartz_Density)
        
        part1_eq = -(thata) * coeff
        part3_eq = ((1 - thata)) * coeff

        term_A = 2 * part1_eq * np.real(val_total) - 2 * part3_eq * np.real(1/q0)
        term_B = 4 * part1_eq * np.imag(val_total) - 4 * part3_eq * np.imag(1/q0)
        
        norm_factor = 1e12 
        
        DF = term_A * norm_factor
        DW = term_B * norm_factor
        return DW, DF

    except Exception:
        return None, None

def model_Kanazawa_line(dn, Quartz_Density, Quartz_Viscosity):
    try:
        q0 = q0_cal(dn)  
        part1_eq = -1/ np.sqrt(Quartz_Viscosity * Quartz_Density)      
        term_A = 2 * part1_eq * np.real(1/q0) 
        term_B = 4 * part1_eq * np.imag(1/q0) 
        
        return term_B * 1e12, term_A * 1e12 # DW, DF
    except: return None, None

def find_cutoff(dn_raw_co, dw_test_co, thata, Liquid_Density, Quartz_Density, Quartz_Viscosity):
    try:
        max_scan = dn_raw_co[-1] 
        calculated_noise_floor = (np.mean(dw_test_co)/1000)

        h_scan_um = np.linspace(0.001, max_scan, 2000)
        h_scan_nm = h_scan_um * 1e3
        h_scan_meters = h_scan_nm * 1e-9 
        
        dn_input = max_scan * 1e-6 
        
        w_soft, _ = dalta_model(dn_input, thata, 5e-9, h_scan_meters, Liquid_Density, Quartz_Density, Quartz_Viscosity)
        w_hard, _ = dalta_model(dn_input, thata, 20e-9, h_scan_meters, Liquid_Density, Quartz_Density, Quartz_Viscosity)

        if w_soft is None or w_hard is None: return None

        diff_array = np.abs(w_soft - w_hard)
        valid_indices = np.where(diff_array > calculated_noise_floor)[0]
        
        if len(valid_indices) > 0:
            return h_scan_nm[valid_indices[0]]
        else:
            return None
    except:
        return None

# ==========================================
# --- 2. Custom Toolbar Class ---
# ==========================================
class CustomVerticalToolbar(NavigationToolbar2Tk):
    toolitems = (
        ('Home', 'Reset original view', 'home', 'home'),
        ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
        ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
    )

    def __init__(self, canvas, window):
        super().__init__(canvas, window)
        
        bg_color = "#404040" 
        self.config(background=bg_color)
        
        self._message_label.pack_forget()

        for widget in self.winfo_children():
            if widget != self._message_label:
                widget.config(background=bg_color)
                widget.pack_forget()
                widget.pack(side="top", pady=5, padx=2, fill="x")

    def set_message(self, s):
        pass

# ==========================================
# --- 3. Window Classes (Graphs & Tools) ---
# ==========================================

class TimeSyncWindow(ctk.CTkToplevel):
    def __init__(self, parent_app):
        super().__init__()
        self.title("Time Synchronization Tool")
        self.geometry("1100x850") 
        self.after(10, lambda: self.state('zoomed'))
        self.parent_app = parent_app
        
        self.df_qcm = parent_app.df_original.copy() if parent_app.df_original is not None else None
        self.df_echem = parent_app.df_echem_original.copy() if parent_app.df_echem_original is not None else None

        self.setup_ui()

        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def setup_ui(self):
        if self.df_qcm is None or self.df_echem is None:
            ctk.CTkLabel(self, text="Please load both QCM-D and EChem Data first.", font=("Arial", 16)).pack(pady=40)
            return

        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        self.qcm_t_col = next((c for c in self.df_qcm.columns if any(t.lower() in c.lower() for t in possible_time_cols)), None)
        if not self.qcm_t_col:
            ctk.CTkLabel(self, text="Could not find Time column in QCM data.").pack(pady=40)
            return
        self.qcm_time = pd.to_numeric(self.df_qcm[self.qcm_t_col], errors='coerce').fillna(0).to_numpy()

        self.f_data = {}
        self.harmonics = []
        for col in self.df_qcm.columns:
            if col.startswith('f') and col[1:].isdigit():
                n = int(col[1:])
                self.harmonics.append(n)
                raw_f = pd.to_numeric(self.df_qcm[col], errors='coerce').fillna(0).to_numpy()
                norm_f = (raw_f - raw_f[0]) / n
                self.f_data[n] = norm_f
        self.harmonics.sort()

        e_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'time/s']
        e_cols = ['Ewe', 'Potential', 'Voltage', '<Ewe/V>', 'Ewe/V']
        q_cols = ['Q', 'Charge', 'C', '(Q-Qo)', '(Q-Qo)/mC', 'Q/mC']
        
        self.ec_t_col = next((c for c in self.df_echem.columns if any(x in c for x in e_time_cols)), None)
        self.ec_e_col = next((c for c in self.df_echem.columns if any(x in c for x in e_cols)), None)
        self.ec_q_col = next((c for c in self.df_echem.columns if any(x in c for x in q_cols)), None)

        if not self.ec_t_col or not self.ec_e_col:
            ctk.CTkLabel(self, text="Could not find Time or Voltage ('E') columns in EChem data.").pack(pady=40)
            return

        self.ec_time = pd.to_numeric(self.df_echem[self.ec_t_col], errors='coerce').fillna(0).to_numpy()
        self.ec_e = pd.to_numeric(self.df_echem[self.ec_e_col], errors='coerce').fillna(0).to_numpy()
        
        if self.ec_q_col:
            self.ec_q = pd.to_numeric(self.df_echem[self.ec_q_col], errors='coerce').fillna(0).to_numpy()
        else:
            self.ec_q = np.zeros_like(self.ec_time)

        avg_f = np.zeros_like(self.qcm_time)
        for n in self.harmonics:
            avg_f += self.f_data[n]
        if len(self.harmonics) > 0:
            avg_f /= len(self.harmonics)
        
        w_len = max(5, len(avg_f) // 100)
        sm_f = np.convolve(avg_f, np.ones(w_len)/w_len, mode='same')
        prom = (np.max(sm_f) - np.min(sm_f)) * 0.05
        dist = max(10, len(sm_f) // 20)
        self.f_valleys, _ = find_peaks(-sm_f, prominence=prom, distance=dist)

        w_len_q = max(5, len(self.ec_q) // 100)
        sm_q = np.convolve(self.ec_q, np.ones(w_len_q)/w_len_q, mode='same')
        prom_q = (np.max(sm_q) - np.min(sm_q)) * 0.05
        dist_q = max(10, len(sm_q) // 20)
        self.q_valleys, _ = find_peaks(-sm_q, prominence=prom_q, distance=dist_q)

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="both", expand=True)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", side="bottom", pady=15)

        max_range = 1000.0 
        if len(self.ec_time) > 0 and len(self.qcm_time) > 0:
            max_range = float(max(np.max(self.ec_time), np.max(self.qcm_time)))

        self.offset_var = ctk.DoubleVar(value=self.parent_app.current_time_offset)
        
        self.btn_auto = ctk.CTkButton(bottom_frame, text="Auto Sync", fg_color="#FF8C00", hover_color="#E67E22", command=self.auto_sync_execute, font=("Arial", 12, "bold"))
        self.btn_auto.pack(side="left", padx=20)
        
        center_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        center_frame.pack(side="left", fill="x", expand=True, padx=10)
        
        ctk.CTkLabel(center_frame, text="Offset (s):", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        self.entry_offset = ctk.CTkEntry(center_frame, textvariable=self.offset_var, width=80)
        self.entry_offset.pack(side="left", padx=5)
        self.entry_offset.bind("<Return>", lambda e: self.update_plot())
        self.entry_offset.bind("<FocusOut>", lambda e: self.update_plot())

        self.slider_offset = ctk.CTkSlider(center_frame, from_=-max_range, to=max_range, variable=self.offset_var, command=self.slider_moved)
        self.slider_offset.pack(side="left", fill="x", expand=True, padx=20)

        self.btn_apply = ctk.CTkButton(bottom_frame, text="Apply Offset & Crop", fg_color="green", command=self.apply_and_crop, font=("Arial", 12, "bold"))
        self.btn_apply.pack(side="right", padx=20)

        self.fig, self.ax1 = plt.subplots(figsize=(10, 8))
        self.ax2 = self.ax1.twinx()
        self.ax3 = self.ax1.twinx()
        
        self.ax3.spines['right'].set_position(('outward', 60))
        self.fig.patch.set_facecolor('white')
        self.ax1.set_facecolor('white')
        self.ax1.grid(True, linestyle='--', color='gray', alpha=0.5)
        
        for ax in [self.ax1, self.ax2, self.ax3]:
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.f_colors = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}

        self.e_line = None
        self.q_line = None
        
        self.draw_initial_plot()
        self.update_plot()

    def auto_sync_execute(self):
        if len(self.f_valleys) > 0 and len(self.q_valleys) > 0:
            t_f_first = self.qcm_time[self.f_valleys[0]]
            t_q_first = self.ec_time[self.q_valleys[0]]
            optimal_offset = t_f_first - t_q_first
            self.offset_var.set(optimal_offset)
            self.update_plot()
        else:
            messagebox.showwarning("Auto Sync", "Could not detect clear valleys in both F and Q data to perform Auto Sync.")

    def draw_initial_plot(self):
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        self.ax3.spines['right'].set_position(('outward', 60))
        
        self.ax1.set_xlabel("Time [s]", color='black', fontsize=10)
        self.ax1.set_ylabel(r"$\Delta f_n / n$ [Hz]", color='black', fontsize=10)
        
        self.ax2.set_ylabel("E (V)", color='#555555', fontsize=10)
        self.ax3.set_ylabel("Q (mC)", color='purple', fontsize=10)
        
        self.ax2.yaxis.set_label_position("right")
        self.ax2.yaxis.tick_right()
        self.ax3.yaxis.set_label_position("right")
        self.ax3.yaxis.tick_right()

        self.ax1.tick_params(axis='x', colors='black')
        self.ax1.tick_params(axis='y', colors='black')
        self.ax2.tick_params(axis='y', colors='#555555')
        self.ax3.tick_params(axis='y', colors='purple')

        for n in self.harmonics:
            color = self.f_colors.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
            self.ax1.plot(self.qcm_time, self.f_data[n], color=color, linewidth=1.2, alpha=0.8, label=f"f{n}")
            
        self.e_line, = self.ax2.plot(self.ec_time, self.ec_e, color='grey', linestyle='-', linewidth=2.0, alpha=0.6, label='E')
        self.q_line, = self.ax3.plot(self.ec_time, self.ec_q, color='purple', linestyle='--', linewidth=1.5, alpha=0.8, label='Q')

        lines1, labels1 = self.ax1.get_legend_handles_labels()
        lines2, labels2 = self.ax2.get_legend_handles_labels()
        lines3, labels3 = self.ax3.get_legend_handles_labels()
        self.ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc="upper right", fontsize='small')

        if len(self.qcm_time) > 0:
            qcm_min = np.min(self.qcm_time)
            qcm_max = np.max(self.qcm_time)
            padding = (qcm_max - qcm_min) * 0.05
            self.ax1.set_xlim(qcm_min - padding, qcm_max + padding)

        self.fig.tight_layout()
        self.fig.subplots_adjust(right=0.85)
        self.canvas.draw()

    def slider_moved(self, val):
        self.update_plot()

    def update_plot(self):
        try: offset = self.offset_var.get()
        except: offset = 0.0
            
        shifted_ec_time = self.ec_time + offset
        
        if self.e_line: self.e_line.set_xdata(shifted_ec_time)
        if self.q_line: self.q_line.set_xdata(shifted_ec_time)

        self.ax2.relim(); self.ax2.autoscale_view(scalex=False, scaley=True)
        self.ax3.relim(); self.ax3.autoscale_view(scalex=False, scaley=True)
        self.canvas.draw_idle()

    def apply_and_crop(self):
        offset = self.offset_var.get()
        self.parent_app.current_time_offset = offset
        
        df_new = self.parent_app.df_original.copy()
        df_new = df_new[df_new[self.qcm_t_col] >= offset].reset_index(drop=True)
        
        df_ec_new = self.parent_app.df_echem_original.copy()
        df_ec_new[self.ec_t_col] = df_ec_new[self.ec_t_col] + offset
        
        self.parent_app.df = df_new
        self.parent_app.df_echem = df_ec_new
        
        self.parent_app.baseline_indices = None
        self.parent_app.invalidate_full_run()
        
        self.parent_app.log(f">>> Time Sync Applied. Offset: {offset:.2f}s. Rows remaining: {len(df_new)}")
        self.parent_app.lbl_sync_ref.configure(text=f"Ref Offset: {offset:.2f}s")
        self.destroy()

# ==========================================
# --- BASELINE SELECTION WINDOW ---
# ==========================================

class BaselineSelectionWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, df_subset, harmonics, on_confirm_callback, default_idx):
        super().__init__()
        self.title("Select Baseline Point (Double-Click on the graph)")
        self.geometry("1100x850")
        self.after(10, lambda: self.state('zoomed'))
        
        self.df = df_subset
        self.harmonics = harmonics
        self.on_confirm_callback = on_confirm_callback
        self.selected_idx = default_idx
        
        # Pull global zeros for accurate local representation
        self.global_f0 = {}
        self.global_d0 = {}
        if parent_app.working_df is not None and not parent_app.working_df.empty:
            for n in harmonics:
                col_f = f"f{n}"
                if col_f in parent_app.working_df.columns:
                    self.global_f0[n] = parent_app.working_df[col_f].iloc[0]
                col_d = f"D{n}" if f"D{n}" in parent_app.working_df.columns else f"d{n}"
                if col_d in parent_app.working_df.columns:
                    self.global_d0[n] = parent_app.working_df[col_d].iloc[0]
        
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        self.t_col = next((c for c in self.df.columns if any(t.lower() in c.lower() for t in possible_time_cols)), None)
        
        if self.t_col:
            self.x_data = pd.to_numeric(self.df[self.t_col], errors='coerce').fillna(0).to_numpy()
        else:
            self.x_data = self.df.index.to_numpy()
            
        self.lines_data_f = {}
        self.lines_data_d = {}
        self.vline_f = None
        self.vline_d = None
        self.dot_f = None
        self.dot_d = None

        self.setup_ui()
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def setup_ui(self):
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        lbl = ctk.CTkLabel(graph_frame, text="Double-click on either the Dissipation or Frequency graph to set the baseline point.\nThen click Confirm.", font=("Arial", 16, "bold"), text_color="#f1c40f")
        lbl.pack(pady=10)

        self.fig, (self.ax_d, self.ax_f) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.patch.set_facecolor('white')
        self.fig.subplots_adjust(right=0.85, bottom=0.1, left=0.1, hspace=0.1) 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", side="bottom", pady=15)

        self.lbl_selected = ctk.CTkLabel(bottom_frame, text="Selected Point: N/A", font=("Arial", 14, "bold"))
        self.lbl_selected.pack(side="left", padx=20)

        btn_confirm = ctk.CTkButton(bottom_frame, text="Confirm Baseline", fg_color="green", hover_color="#27AE60", command=self.confirm, font=("Arial", 14, "bold"))
        btn_confirm.pack(side="right", padx=20)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.plot_graphs()

    def plot_graphs(self):
        self.ax_f.clear()
        self.ax_d.clear()
        
        self.ax_f.set_ylabel(r"$\Delta f_n / n$ [Hz]", fontsize=11)
        self.ax_d.set_ylabel(r"$\Delta D_n$ [ppm]", fontsize=11)
        
        if self.t_col:
            self.ax_f.set_xlabel("Time [s]", fontsize=11)
            
        self.ax_f.grid(True, linestyle='--', alpha=0.5)
        self.ax_d.grid(True, linestyle='--', alpha=0.5)
        
        f_colors_fixed = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}
        d_colors_fixed = {3: "#FF2C14", 5: "#FD5328", 7: "#FD753A", 9: "#FAA56D", 11: "#FBB261", 13: "#FBC873"}
        
        for n in self.harmonics:
            col_name_f = f"f{n}"
            if col_name_f in self.df.columns:
                raw_f = pd.to_numeric(self.df[col_name_f], errors='coerce').fillna(0).to_numpy()
                f0 = self.global_f0.get(n, raw_f[0] if len(raw_f) > 0 else 0)
                norm_f = (raw_f - f0) / n
                c_f = f_colors_fixed.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_f.plot(self.x_data, norm_f, color=c_f, linewidth=0.8, alpha=0.8, marker='o', markersize=1, label=f"f{n}")
                self.lines_data_f[n] = norm_f
                
            col_name_d = f"D{n}"
            if col_name_d not in self.df.columns: col_name_d = f"d{n}"
            if col_name_d in self.df.columns:
                raw_d = pd.to_numeric(self.df[col_name_d], errors='coerce').fillna(0).to_numpy()
                d0 = self.global_d0.get(n, raw_d[0] if len(raw_d) > 0 else 0)
                delta_d = (raw_d - d0) 
                c_d = d_colors_fixed.get(n, plt.cm.YlOrRd(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_d.plot(self.x_data, delta_d, color=c_d, linewidth=0.8, alpha=0.8, marker='o', markersize=1, label=rf"$\Delta D_{{{n}}}$")
                self.lines_data_d[n] = delta_d
                
        self.ax_f.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        self.ax_d.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        
        self.update_selection_visuals()

    def update_selection_visuals(self):
        if self.selected_idx in self.df.index:
            loc = self.df.index.get_loc(self.selected_idx)
            x_val = self.x_data[loc]
            
            if self.vline_f: self.vline_f.remove()
            if self.vline_d: self.vline_d.remove()
            if self.dot_f: self.dot_f.remove()
            if self.dot_d: self.dot_d.remove()
            
            self.vline_f = self.ax_f.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
            self.vline_d = self.ax_d.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
            
            y_val_f = 0
            y_val_d = 0
            
            if len(self.harmonics) > 0:
                first_n = self.harmonics[0]
                if first_n in self.lines_data_f:
                    y_val_f = self.lines_data_f[first_n][loc]
                if first_n in self.lines_data_d:
                    y_val_d = self.lines_data_d[first_n][loc]
                    
            self.dot_f, = self.ax_f.plot(x_val, y_val_f, 'ro', markersize=8, markeredgecolor='black', zorder=5)
            self.dot_d, = self.ax_d.plot(x_val, y_val_d, 'ro', markersize=8, markeredgecolor='black', zorder=5)
            
            time_str = f"{x_val:.1f}s" if self.t_col else str(x_val)
            self.lbl_selected.configure(text=f"Selected Baseline -> Time: {time_str} | Original Row Index: {self.selected_idx}")
            self.canvas.draw()

    def on_click(self, event):
        if not event.dblclick: return
        if event.inaxes not in [self.ax_f, self.ax_d]: return
        if event.xdata is not None:
            distances = np.abs(self.x_data - event.xdata)
            nearest_loc = distances.argmin()
            self.selected_idx = self.df.index[nearest_loc]
            self.update_selection_visuals()

    def confirm(self):
        self.destroy()
        self.on_confirm_callback(self.selected_idx)

# ==========================================
# --- MANUAL SEGMENTATION WINDOW ---
# ==========================================

class ManualSegmentationWindow(ctk.CTkToplevel):
    def __init__(self, parent_app):
        super().__init__()
        self.title("Manual Segmentation Selection")
        self.geometry("1100x850")
        self.after(10, lambda: self.state('zoomed'))
        
        self.parent_app = parent_app
        self.df = parent_app.working_df.copy()
        self.harmonics = parent_app.current_harmonics
        self.split_lines = []
        
        self.global_f0 = {}
        self.global_d0 = {}
        if parent_app.working_df is not None and not parent_app.working_df.empty:
            for n in self.harmonics:
                col_f = f"f{n}"
                if col_f in parent_app.working_df.columns:
                    self.global_f0[n] = parent_app.working_df[col_f].iloc[0]
                col_d = f"D{n}" if f"D{n}" in parent_app.working_df.columns else f"d{n}"
                if col_d in parent_app.working_df.columns:
                    self.global_d0[n] = parent_app.working_df[col_d].iloc[0]
        
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        self.t_col = next((c for c in self.df.columns if any(t.lower() in c.lower() for t in possible_time_cols)), None)
        
        if self.t_col:
            self.x_data = pd.to_numeric(self.df[self.t_col], errors='coerce').fillna(0).to_numpy()
        else:
            self.x_data = self.df.index.to_numpy()
            
        self.setup_ui()
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def setup_ui(self):
        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        instruction_text = (
            "Double-click on the graph to add a red vertical cut-line.\n"
            "Double-click on an existing line to remove it.\n"
            "The system will perform separate optimization for each segment between the lines."
        )
        lbl = ctk.CTkLabel(graph_frame, text=instruction_text, font=("Arial", 15, "bold"), text_color="#E74C3C")
        lbl.pack(pady=10)

        self.fig, (self.ax_d, self.ax_f) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        self.fig.patch.set_facecolor('white')
        self.fig.subplots_adjust(right=0.85, bottom=0.1, left=0.1, hspace=0.1) 
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", side="bottom", pady=15)

        btn_clear = ctk.CTkButton(bottom_frame, text="Clear All Lines", fg_color="#E74C3C", hover_color="#C0392B", command=self.clear_lines, font=("Arial", 14, "bold"))
        btn_clear.pack(side="left", padx=40)

        btn_confirm = ctk.CTkButton(bottom_frame, text="Confirm Segments", fg_color="green", hover_color="#27AE60", command=self.confirm, font=("Arial", 14, "bold"))
        btn_confirm.pack(side="right", padx=40)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.plot_graphs()

    def plot_graphs(self):
        self.ax_f.clear()
        self.ax_d.clear()
        
        self.ax_f.set_ylabel(r"$\Delta f_n / n$ [Hz]", fontsize=11)
        self.ax_d.set_ylabel(r"$\Delta D_n$ [ppm]", fontsize=11)
        
        if self.t_col:
            self.ax_f.set_xlabel("Time [s]", fontsize=11)
            
        self.ax_f.grid(True, linestyle='--', alpha=0.5)
        self.ax_d.grid(True, linestyle='--', alpha=0.5)
        
        f_colors_fixed = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}
        d_colors_fixed = {3: "#FF2C14", 5: "#FD5328", 7: "#FD753A", 9: "#FAA56D", 11: "#FBB261", 13: "#FBC873"}
        
        for n in self.harmonics:
            col_name_f = f"f{n}"
            if col_name_f in self.df.columns:
                raw_f = pd.to_numeric(self.df[col_name_f], errors='coerce').fillna(0).to_numpy()
                f0 = self.global_f0.get(n, raw_f[0] if len(raw_f) > 0 else 0)
                norm_f = (raw_f - f0) / n
                c_f = f_colors_fixed.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_f.plot(self.x_data, norm_f, color=c_f, linewidth=0.8, alpha=0.8, marker='o', markersize=1, label=f"f{n}")
                
            col_name_d = f"D{n}"
            if col_name_d not in self.df.columns: col_name_d = f"d{n}"
            if col_name_d in self.df.columns:
                raw_d = pd.to_numeric(self.df[col_name_d], errors='coerce').fillna(0).to_numpy()
                d0 = self.global_d0.get(n, raw_d[0] if len(raw_d) > 0 else 0)
                delta_d = (raw_d - d0) 
                c_d = d_colors_fixed.get(n, plt.cm.YlOrRd(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_d.plot(self.x_data, delta_d, color=c_d, linewidth=0.8, alpha=0.8, marker='o', markersize=1, label=rf"$\Delta D_{{{n}}}$")
                
        self.ax_f.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        self.ax_d.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
        
        # --- Restore saved lines from memory ---
        for x_val in self.parent_app.saved_manual_splits:
            lf = self.ax_f.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
            ld = self.ax_d.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
            self.split_lines.append({'x': x_val, 'line_f': lf, 'line_d': ld})
            
        self.canvas.draw()

    def on_click(self, event):
        if not event.dblclick: return
        if event.inaxes not in [self.ax_f, self.ax_d]: return
        if event.xdata is not None:
            x_click = event.xdata
            x_range = self.ax_f.get_xlim()
            tolerance = (x_range[1] - x_range[0]) * 0.015 

            clicked_line = None
            for line_data in self.split_lines:
                if abs(line_data['x'] - x_click) < tolerance:
                    clicked_line = line_data
                    break

            if clicked_line:
                clicked_line['line_f'].remove()
                clicked_line['line_d'].remove()
                self.split_lines.remove(clicked_line)
                self.canvas.draw()
            else:
                lf = self.ax_f.axvline(x=x_click, color='red', linestyle='--', linewidth=2.0)
                ld = self.ax_d.axvline(x=x_click, color='red', linestyle='--', linewidth=2.0)
                self.split_lines.append({'x': x_click, 'line_f': lf, 'line_d': ld})
                self.canvas.draw()

    def clear_lines(self):
        for line_data in self.split_lines:
            line_data['line_f'].remove()
            line_data['line_d'].remove()
        self.split_lines = []
        self.canvas.draw()

    def confirm(self):
        x_vals = sorted([line['x'] for line in self.split_lines])
        
        # --- Save confirmed lines to App memory ---
        self.parent_app.saved_manual_splits = x_vals
        
        split_indices = [0]
        for x in x_vals:
            idx = (np.abs(self.x_data - x)).argmin()
            split_indices.append(idx)
        split_indices.append(len(self.x_data) - 1)
        
        split_indices = sorted(list(set(split_indices)))
        
        custom_segments = []
        for i in range(len(split_indices) - 1):
            custom_segments.append((split_indices[i], split_indices[i+1]))
            
        if not custom_segments:
            custom_segments = [(0, len(self.x_data)-1)]
            
        self.destroy()
        
        if self.parent_app.dynamic_window is None or not self.parent_app.dynamic_window.winfo_exists():
            self.parent_app.dynamic_window = DynamicParamsWindow(
                self.parent_app, 
                run_callback=lambda config: self.parent_app.run_full_process(dynamic_config=config, first_cycle_only=False),
                custom_segments=custom_segments
            )
            self.parent_app.dynamic_window.lift()


class OptimizationManualSegmentationWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, df_override=None, title_suffix="", save_key="global", on_confirm_callback=None, global_f0=None, global_d0=None):
        super().__init__()
        self.title("Manual Segmentation Selection")
        self.geometry("1300x950")
        self.after(10, lambda: self.state('zoomed'))

        self.parent_app = parent_app
        self.df = df_override.copy() if df_override is not None else parent_app.working_df.copy()
        self.harmonics = parent_app.current_harmonics
        self.save_key = save_key
        self.on_confirm_callback = on_confirm_callback
        self.title_suffix = title_suffix
        self.split_lines = []
        self.global_f0 = global_f0 if global_f0 is not None else {}
        self.global_d0 = global_d0 if global_d0 is not None else {}

        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', '׳–׳׳', 'Time [sec]', 'AbsTime', 'RelTime']
        self.t_col = next((c for c in self.df.columns if any(t.lower() in c.lower() for t in possible_time_cols)), None)
        if self.t_col:
            self.x_data = pd.to_numeric(self.df[self.t_col], errors='coerce').fillna(0).to_numpy()
        else:
            self.x_data = self.df.index.to_numpy()

        self.y_height_abs = pd.to_numeric(self.df["Graph_h_nm"], errors='coerce').to_numpy() if "Graph_h_nm" in self.df.columns else np.full(len(self.df), np.nan)
        self.y_qsi = pd.to_numeric(self.df["Graph_xi_nm"], errors='coerce').to_numpy() if "Graph_xi_nm" in self.df.columns else np.full(len(self.df), np.nan)
        self.y_theo = pd.to_numeric(self.df["Theo_Calibrated_Active"], errors='coerce').to_numpy() if "Theo_Calibrated_Active" in self.df.columns else None
        self.y_voltage = pd.to_numeric(self.df["E_we_V"], errors='coerce').to_numpy() if "E_we_V" in self.df.columns else None

        self.setup_ui()

        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def setup_ui(self):
        title_text = "Manual Segmentation on Optimization Graphs"
        if self.title_suffix:
            title_text += f" | {self.title_suffix}"
        self.title(title_text)

        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        instruction_text = (
            "Double-click on any graph to add a red vertical cut-line.\n"
            "Double-click on an existing red line to remove it.\n"
            "The split will use the current optimization graphs."
        )
        ctk.CTkLabel(graph_frame, text=instruction_text, font=("Arial", 15, "bold"), text_color="#E74C3C").pack(pady=10)

        self.fig, (self.ax_f, self.ax_d, self.ax_h, self.ax_xi) = plt.subplots(4, 1, figsize=(10, 8.2), sharex=True)
        self.fig.patch.set_facecolor('white')
        self.fig.subplots_adjust(right=0.85, bottom=0.09, left=0.1, hspace=0.22)

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y")

        bottom_frame = ctk.CTkFrame(self, fg_color="transparent", height=56)
        bottom_frame.pack(side="bottom", pady=(4, 10), fill="x")
        bottom_frame.pack_propagate(False)

        ctk.CTkButton(
            bottom_frame,
            text="Clear All Lines",
            fg_color="#E74C3C",
            hover_color="#C0392B",
            command=self.clear_lines,
            width=160,
            height=36
        ).pack(side="left", padx=40)

        ctk.CTkButton(
            bottom_frame,
            text="Confirm Segments",
            fg_color="green",
            hover_color="#27AE60",
            command=self.confirm,
            width=160,
            height=36
        ).pack(side="right", padx=40)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.plot_graphs()

    def plot_graphs(self):
        for ax in [self.ax_f, self.ax_d, self.ax_h, self.ax_xi]:
            ax.clear()
            ax.grid(True, linestyle='--', alpha=0.5)

        self.ax_f.set_ylabel(r"$\Delta f_n / n$ [Hz]", fontsize=10)
        self.ax_d.set_ylabel(r"$\Delta D_n$ [ppm]", fontsize=10)
        self.ax_h.set_ylabel(r"$h$ [nm]", fontsize=10)
        self.ax_xi.set_ylabel(r"$\xi$ [nm]", fontsize=10)
        self.ax_xi.set_xlabel("Time [s]" if self.t_col else "Index", fontsize=10)

        f_colors_fixed = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}
        d_colors_fixed = {3: "#FF2C14", 5: "#FD5328", 7: "#FD753A", 9: "#FAA56D", 11: "#FBB261", 13: "#FBC873"}

        self.ax1_twin = None
        if self.y_voltage is not None:
            self.ax1_twin = self.ax_f.twinx()
            self.ax1_twin.set_ylabel("E [V]", color='#333333', fontsize=9)
            self.ax1_twin.plot(self.x_data, self.y_voltage, color='#333333', alpha=0.8, linewidth=1.0, linestyle='-', label='E [V]')
            self.ax1_twin.tick_params(axis='y', labelcolor='#333333')

        for n in self.harmonics:
            col_name_f = f"f{n}"
            if col_name_f in self.df.columns:
                raw_f = pd.to_numeric(self.df[col_name_f], errors='coerce').fillna(0).to_numpy()
                f0 = self.global_f0.get(n, raw_f[0] if len(raw_f) > 0 else 0)
                norm_f = (raw_f - f0) / n
                c_f = f_colors_fixed.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_f.plot(self.x_data, norm_f, color=c_f, linewidth=1.0, alpha=0.8, label=f"f{n}")

            col_name_d = f"D{n}"
            if col_name_d not in self.df.columns:
                col_name_d = f"d{n}"
            if col_name_d in self.df.columns:
                raw_d = pd.to_numeric(self.df[col_name_d], errors='coerce').fillna(0).to_numpy()
                d0 = self.global_d0.get(n, raw_d[0] if len(raw_d) > 0 else 0)
                delta_d = raw_d - d0
                c_d = d_colors_fixed.get(n, plt.cm.YlOrRd(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax_d.plot(self.x_data, delta_d, color=c_d, linewidth=1.0, alpha=0.8, label=rf"$\Delta D_{{{n}}}$")

        if self.y_theo is not None:
            self.ax_f.plot(self.x_data, self.y_theo, color='black', linestyle='-.', linewidth=1.4, label=r'$\Delta f_{theo} (Calib)$')

        self.ax_h.plot(self.x_data, self.y_height_abs, color='#1f77b4', linewidth=1.5, label=r'Total Height ($h_{total}$)')
        self.ax_xi.plot(self.x_data, self.y_qsi, color='#ff7f0e', linewidth=1.5, label=r'Permeability length ($\xi$)')

        lines, labels = self.ax_f.get_legend_handles_labels()
        if self.ax1_twin is not None:
            lines2, labels2 = self.ax1_twin.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        self.ax_f.legend(lines, labels, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize='small')
        self.ax_d.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize='small')
        self.ax_h.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize='small')
        self.ax_xi.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize='small')

        self.split_lines = []
        for x_val in self.parent_app.manual_split_memory.get(self.save_key, []):
            self.add_split_line(x_val)

        self.canvas.draw()

    def add_split_line(self, x_val):
        line_data = {
            'x': x_val,
            'line_f': self.ax_f.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0),
            'line_d': self.ax_d.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0),
            'line_h': self.ax_h.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0),
            'line_xi': self.ax_xi.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
        }
        if self.ax1_twin is not None:
            line_data['line_f_twin'] = self.ax1_twin.axvline(x=x_val, color='red', linestyle='--', linewidth=2.0)
        self.split_lines.append(line_data)

    def remove_split_line(self, line_data):
        for key in ['line_f', 'line_d', 'line_h', 'line_xi', 'line_f_twin']:
            if key in line_data and line_data[key] is not None:
                line_data[key].remove()
        self.split_lines.remove(line_data)

    def on_click(self, event):
        if not event.dblclick or self.toolbar.mode != "":
            return

        valid_axes = [self.ax_f, self.ax_d, self.ax_h, self.ax_xi]
        if self.ax1_twin is not None:
            valid_axes.append(self.ax1_twin)
        if event.inaxes not in valid_axes or event.xdata is None:
            return

        x_click = event.xdata
        x_range = self.ax_xi.get_xlim()
        tolerance = (x_range[1] - x_range[0]) * 0.015

        clicked_line = None
        for line_data in self.split_lines:
            if abs(line_data['x'] - x_click) < tolerance:
                clicked_line = line_data
                break

        if clicked_line is not None:
            self.remove_split_line(clicked_line)
        else:
            self.add_split_line(x_click)
        self.canvas.draw()

    def clear_lines(self):
        for line_data in list(self.split_lines):
            self.remove_split_line(line_data)
        self.canvas.draw()

    def confirm(self):
        x_vals = sorted([line['x'] for line in self.split_lines])
        self.parent_app.manual_split_memory[self.save_key] = x_vals

        split_indices = [int(self.df.index[0])]
        for x in x_vals:
            idx = (np.abs(self.x_data - x)).argmin()
            split_indices.append(int(self.df.index[idx]))
        split_indices.append(int(self.df.index[-1]))
        split_indices = sorted(list(set(split_indices)))

        custom_segments = []
        for i in range(len(split_indices) - 1):
            custom_segments.append((split_indices[i], split_indices[i + 1]))

        if not custom_segments:
            custom_segments = [(int(self.df.index[0]), int(self.df.index[-1]))]

        self.destroy()
        if self.on_confirm_callback is not None:
            self.on_confirm_callback(custom_segments, x_vals)
        else:
            self.parent_app.dynamic_window.destroy()
            self.parent_app.dynamic_window = DynamicParamsWindow(
                self.parent_app, 
                run_callback=lambda config: self.parent_app.run_full_process(dynamic_config=config, first_cycle_only=False),
                custom_segments=custom_segments
            )
            self.parent_app.dynamic_window.lift()

# ==========================================
# --- Global Viscosity Calibration Window ---
# ==========================================

class ViscosityCalibrationWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, target_idx=5):
        super().__init__()
        self.title("Global Viscosity Calibration")
        self.geometry("1100x850") 
        self.after(10, lambda: self.state('zoomed'))
        self.parent_app = parent_app
        self.target_idx = target_idx
        
        self.df = parent_app.df.copy() if parent_app.df is not None else None
        self.harmonics = parent_app.current_harmonics
        
        self.median_f = {}
        self.median_d = {}
        self.is_scanning = False
        self.selection_mode = "manual"
        self._setting_auto_value = False
        self.selection_mode = "manual"
        self._setting_auto_value = False
        self.selection_mode = "manual"
        self._setting_auto_value = False
        self.selection_mode = "manual"
        self._setting_auto_value = False
        
        try:
            start_visc = float(parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].get())
        except:
            start_visc = 0.0032
            
        self.visc_var = ctk.DoubleVar(value=start_visc)

        self.setup_ui()
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

        if self.df is not None:
            self.after(200, self.run_initial_scan)

    def setup_ui(self):
        if self.df is None:
            ctk.CTkLabel(self, text="Please load QCM-D Data first.", font=("Arial", 16)).pack(pady=40)
            return

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="both", expand=True)
        
        self.lbl_status = ctk.CTkLabel(top_frame, text="Initializing...", font=("Arial", 14, "bold"), text_color="#FF8C00")
        self.lbl_status.pack(pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 7))
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, hspace=0.3)
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", side="bottom", pady=15)

        self.btn_auto = ctk.CTkButton(bottom_frame, text="Auto Calibrate Viscosity", fg_color="#3498db", hover_color="#2980b9", command=self.auto_calibrate, font=("Arial", 12, "bold"))
        self.btn_auto.pack(side="left", padx=20)
        
        center_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        center_frame.pack(side="left", fill="x", expand=True, padx=10)
        
        ctk.CTkLabel(center_frame, text="Viscosity (Pa·s):", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        self.entry_visc = ctk.CTkEntry(center_frame, textvariable=self.visc_var, width=80)
        self.entry_visc.pack(side="left", padx=5)
        self.entry_visc.bind("<Return>", lambda e: self.mark_manual_and_update())
        self.entry_visc.bind("<FocusOut>", lambda e: self.mark_manual_and_update())

        self.slider_visc = ctk.CTkSlider(center_frame, from_=0.0001, to=0.05, variable=self.visc_var, command=self.slider_moved)
        self.slider_visc.pack(side="left", fill="x", expand=True, padx=20)

        self.btn_apply = ctk.CTkButton(bottom_frame, text="Apply Viscosity", fg_color="green", command=self.apply_viscosity, font=("Arial", 12, "bold"))
        self.btn_apply.pack(side="right", padx=20)

    def run_initial_scan(self):
        self.is_scanning = True
        self.btn_apply.configure(state="disabled")
        
        p = self.parent_app.get_params()
        if not p:
            self.lbl_status.configure(text="Parameter error. Scan aborted.", text_color="red")
            self.is_scanning = False
            self.btn_apply.configure(state="normal")
            return

        target_idx = self.target_idx

        for n in self.harmonics:
            col_f = f"f{n}"
            col_d = f"D{n}" if f"D{n}" in self.df.columns else f"d{n}"
            
            self.median_f[n] = float(self.df.loc[target_idx, col_f])
            self.median_d[n] = float(self.df.loc[target_idx, col_d])
            
        self.lbl_status.configure(text=f"Scan Complete. Using Index {target_idx} for baseline.", text_color="green")
        self.is_scanning = False
        self.btn_apply.configure(state="normal")
        self.update_plot()

    def auto_calibrate(self):
        if not self.median_f or not self.median_d:
            messagebox.showwarning("Warning", "Please wait for the baseline scan to complete.", parent=self)
            return
            
        p = self.parent_app.get_params()
        if not p: return

        self.lbl_status.configure(text="Running Auto Calibration...", text_color="#3498db")
        self.update_idletasks()

        target_df = []
        target_dw = []
        
        for n in self.harmonics:
            f_air = p["air_f"][n]
            d_air = p["air_d"][n]
            med_f = self.median_f[n]
            med_d = self.median_d[n]
            
            delta_f = med_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - 0 
            calc_df = num_f / denominator
            
            term_w_liq = (med_d * 1e-6) * (med_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            
            target_df.append(calc_df * 1e9)
            target_dw.append(calc_dw * 1e9)
            
        target_df = np.array(target_df)
        target_dw = np.array(target_dw)
        
        weights_list = [10 if k < 3 else 1.0 for k in range(len(self.harmonics))]
        f_weights_arr = np.array(weights_list)

        def objective_function(params_opt):
            visc_cand = params_opt[0]
            
            row_dn_um = []
            for n in self.harmonics:
                f_air = p["air_f"][n]
                val_dn = 1e6 * np.sqrt(visc_cand / (np.pi * p["rho_liq"] * f_air))
                row_dn_um.append(val_dn)
            row_dn_meters = np.array(row_dn_um) * 1e-6
            
            w_kana, f_kana = model_Kanazawa_line(row_dn_meters, p["rho_quartz"], p["visc_quartz"])
            if w_kana is None: return 1e9
            
            error_w = np.sum((w_kana - target_dw)**2)
            error_f = np.sum(((f_kana - target_df)**2) * f_weights_arr)
            return (100 * error_w) + error_f

        bounds = [(0.0001, 0.1)] 
        result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=1000, popsize=15, tol=1e-5)
        
        if result.success:
            best_visc = result.x[0]
            self._setting_auto_value = True
            self._setting_auto_value = True
            self.visc_var.set(best_visc)
            self._setting_auto_value = False
            self.selection_mode = "auto"
            self._setting_auto_value = False
            self.selection_mode = "auto"
            self.update_plot()
            self.lbl_status.configure(text=f"Calibration Successful: {best_visc:.5f} Pa·s", text_color="green")
        else:
            self.lbl_status.configure(text="Calibration Failed.", text_color="red")
            messagebox.showerror("Error", "Optimization failed to converge.", parent=self)

    def slider_moved(self, val):
        if not self.is_scanning:
            if not self._setting_auto_value:
                self.selection_mode = "manual"
            self.update_plot()

    def mark_manual_and_update(self):
        if not self._setting_auto_value:
            self.selection_mode = "manual"
        self.update_plot()

    def update_plot(self):
        if not self.median_f or not self.median_d: return
        
        try: current_visc = self.visc_var.get()
        except: return
            
        p = self.parent_app.get_params()
        if not p: return
        
        exp_df = []; exp_dw = []; exp_dn = []
        
        for n in self.harmonics:
            f_air = p["air_f"][n]
            d_air = p["air_d"][n]
            
            val_dn = 1e6 * np.sqrt(current_visc / (np.pi * p["rho_liq"] * f_air))
            exp_dn.append(val_dn)
            
            med_f = self.median_f[n]
            med_d = self.median_d[n]
            
            delta_f = med_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - 0 
            calc_df = num_f / denominator
            
            term_w_liq = (med_d * 1e-6) * (med_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            
            exp_df.append(calc_df * 1e9)
            exp_dw.append(calc_dw * 1e9)
            
        dn_smooth_m = np.linspace(0.001, max(exp_dn)*1.5 if exp_dn else 0.3, 500) * 1e-6
        kana_w, kana_f = model_Kanazawa_line(dn_smooth_m, p["rho_quartz"], p["visc_quartz"])
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(dn_smooth_m*1e6, kana_f, 'k--', label='Kanazawa (Model Baseline)', alpha=0.7)
        self.ax1.plot(exp_dn, exp_df, 'bo', label=r'Exp Median ($\Delta f_{calib}$)')
        self.ax1.set_ylabel(r'$\Delta f / n$ [Hz]')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.plot(dn_smooth_m*1e6, kana_w, 'k--', label='Kanazawa (Model Baseline)', alpha=0.7)
        self.ax2.plot(exp_dn, exp_dw, 'ro', label='Exp Median')
        self.ax2.set_ylabel(r'$\Delta W$ [ppm]')
        self.ax2.set_xlabel(r'Penetration Depth $\delta$ [$\mu m$]')
        self.ax2.grid(True)
        self.ax2.legend()
        
        self.canvas.draw_idle()

    def apply_viscosity(self):
        new_visc = self.visc_var.get()
        self.parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].delete(0, 'end')
        self.parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].insert(0, f"{new_visc:.5f}")
        self.parent_app.invalidate_full_run()
        self.parent_app.log(f">>> Viscosity updated to {new_visc:.5f} Pa·s via Calibration.")
        self.destroy()

# ==========================================
# --- Cycle Specific Viscosity Calibration Window ---
# ==========================================

class CycleViscosityCalibrationWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, start_idx, target_idx, target_entry, target_btn, group_id=None):
        super().__init__()
        self.title(f"Cycle Viscosity Calibration (Based on Row {start_idx})")
        self.geometry("1100x850") 
        self.after(10, lambda: self.state('zoomed'))
        self.parent_app = parent_app
        self.start_idx = start_idx
        self.target_idx = target_idx
        self.target_entry = target_entry
        self.target_btn = target_btn
        self.group_id = group_id
        
        self.df = parent_app.working_df.copy() if parent_app.working_df is not None else None
        self.harmonics = parent_app.current_harmonics
        
        self.median_f = {}
        self.median_d = {}
        self.is_scanning = False
        self.selection_mode = "manual"
        self._setting_auto_value = False
        
        try:
            start_visc = float(parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].get())
        except:
            start_visc = 0.0032
            
        self.visc_var = ctk.DoubleVar(value=start_visc)

        self.setup_ui()
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

        if self.df is not None:
            self.after(200, self.run_initial_scan)

    def setup_ui(self):
        if self.df is None:
            ctk.CTkLabel(self, text="No Data found.", font=("Arial", 16)).pack(pady=40)
            return

        top_frame = ctk.CTkFrame(self, fg_color="transparent")
        top_frame.pack(fill="both", expand=True)
        
        self.lbl_status = ctk.CTkLabel(top_frame, text="Initializing...", font=("Arial", 14, "bold"), text_color="#FF8C00")
        self.lbl_status.pack(pady=5)

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 7))
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95, hspace=0.3)
        self.fig.patch.set_facecolor('white')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        bottom_frame = ctk.CTkFrame(self)
        bottom_frame.pack(fill="x", side="bottom", pady=15)

        self.btn_auto = ctk.CTkButton(bottom_frame, text="Auto Calibrate Viscosity", fg_color="#3498db", hover_color="#2980b9", command=self.auto_calibrate, font=("Arial", 12, "bold"))
        self.btn_auto.pack(side="left", padx=20)
        
        center_frame = ctk.CTkFrame(bottom_frame, fg_color="transparent")
        center_frame.pack(side="left", fill="x", expand=True, padx=10)
        
        ctk.CTkLabel(center_frame, text="Viscosity (Pa·s):", font=("Arial", 12, "bold")).pack(side="left", padx=10)
        self.entry_visc = ctk.CTkEntry(center_frame, textvariable=self.visc_var, width=80)
        self.entry_visc.pack(side="left", padx=5)
        self.entry_visc.bind("<Return>", lambda e: self.mark_manual_and_update())
        self.entry_visc.bind("<FocusOut>", lambda e: self.mark_manual_and_update())

        self.slider_visc = ctk.CTkSlider(center_frame, from_=0.0001, to=0.05, variable=self.visc_var, command=self.slider_moved)
        self.slider_visc.pack(side="left", fill="x", expand=True, padx=20)

        self.btn_apply = ctk.CTkButton(bottom_frame, text="Apply to Cycle", fg_color="green", command=self.apply_viscosity, font=("Arial", 12, "bold"))
        self.btn_apply.pack(side="right", padx=20)

    def run_initial_scan(self):
        self.is_scanning = True
        self.btn_apply.configure(state="disabled")
        
        p = self.parent_app.get_params()
        if not p: 
            self.lbl_status.configure(text="Parameter error. Scan aborted.", text_color="red")
            self.is_scanning = False
            self.btn_apply.configure(state="normal")
            return

        target_idx = self.target_idx

        for n in self.harmonics:
            col_f = f"f{n}"
            col_d = f"D{n}" if f"D{n}" in self.df.columns else f"d{n}"
            
            self.median_f[n] = float(self.df.loc[target_idx, col_f])
            self.median_d[n] = float(self.df.loc[target_idx, col_d])
            
        self.lbl_status.configure(text=f"Scan Complete. Using Index {target_idx} for cycle baseline.", text_color="green")
        self.is_scanning = False
        self.btn_apply.configure(state="normal")
        self.update_plot()

    def auto_calibrate(self):
        if not self.median_f or not self.median_d:
            messagebox.showwarning("Warning", "Please wait for the baseline scan to complete.", parent=self)
            return
            
        p = self.parent_app.get_params()
        if not p: return

        self.lbl_status.configure(text="Running Auto Calibration...", text_color="#3498db")
        self.update_idletasks()

        target_df = []
        target_dw = []
        
        for n in self.harmonics:
            f_air = p["air_f"][n]
            d_air = p["air_d"][n]
            med_f = self.median_f[n]
            med_d = self.median_d[n]
            
            delta_f = med_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - 0 
            calc_df = num_f / denominator
            
            term_w_liq = (med_d * 1e-6) * (med_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            
            target_df.append(calc_df * 1e9)
            target_dw.append(calc_dw * 1e9)
            
        target_df = np.array(target_df)
        target_dw = np.array(target_dw)
        
        weights_list = [10 if k < 3 else 1.0 for k in range(len(self.harmonics))]
        f_weights_arr = np.array(weights_list)

        def objective_function(params_opt):
            visc_cand = params_opt[0]
            
            row_dn_um = []
            for n in self.harmonics:
                f_air = p["air_f"][n]
                val_dn = 1e6 * np.sqrt(visc_cand / (np.pi * p["rho_liq"] * f_air))
                row_dn_um.append(val_dn)
            row_dn_meters = np.array(row_dn_um) * 1e-6
            
            w_kana, f_kana = model_Kanazawa_line(row_dn_meters, p["rho_quartz"], p["visc_quartz"])
            if w_kana is None: return 1e9
            
            error_w = np.sum((w_kana - target_dw)**2)
            error_f = np.sum(((f_kana - target_df)**2) * f_weights_arr)
            return (100 * error_w) + error_f

        bounds = [(0.0001, 0.1)] 
        result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=500, popsize=10, tol=1e-3)
        
        if result.success:
            best_visc = result.x[0]
            self._setting_auto_value = True
            self.visc_var.set(best_visc)
            self._setting_auto_value = False
            self.selection_mode = "auto"
            self.update_plot()
            self.lbl_status.configure(text=f"Calibration Successful: {best_visc:.5f} Pa·s", text_color="green")
        else:
            self.lbl_status.configure(text="Calibration Failed.", text_color="red")
            messagebox.showerror("Error", "Optimization failed to converge.", parent=self)

    def slider_moved(self, val):
        if not self.is_scanning:
            if not self._setting_auto_value:
                self.selection_mode = "manual"
            self.update_plot()

    def mark_manual_and_update(self):
        if not self._setting_auto_value:
            self.selection_mode = "manual"
        self.update_plot()

    def update_plot(self):
        if not self.median_f or not self.median_d: return
        
        try: current_visc = self.visc_var.get()
        except: return
            
        p = self.parent_app.get_params()
        if not p: return
        
        exp_df = []; exp_dw = []; exp_dn = []
        
        for n in self.harmonics:
            f_air = p["air_f"][n]
            d_air = p["air_d"][n]
            
            val_dn = 1e6 * np.sqrt(current_visc / (np.pi * p["rho_liq"] * f_air))
            exp_dn.append(val_dn)
            
            med_f = self.median_f[n]
            med_d = self.median_d[n]
            
            delta_f = med_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - 0 
            calc_df = num_f / denominator
            
            term_w_liq = (med_d * 1e-6) * (med_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            
            exp_df.append(calc_df * 1e9)
            exp_dw.append(calc_dw * 1e9)
            
        dn_smooth_m = np.linspace(0.001, max(exp_dn)*1.5 if exp_dn else 0.3, 500) * 1e-6
        kana_w, kana_f = model_Kanazawa_line(dn_smooth_m, p["rho_quartz"], p["visc_quartz"])
        
        self.ax1.clear()
        self.ax2.clear()
        
        self.ax1.plot(dn_smooth_m*1e6, kana_f, 'k--', label='Kanazawa (Model Baseline)', alpha=0.7)
        self.ax1.plot(exp_dn, exp_df, 'bo', label=r'Exp Median ($\Delta f_{calib}$)')
        self.ax1.set_ylabel(r'$\Delta f / n$ [Hz]')
        self.ax1.grid(True)
        self.ax1.legend()
        
        self.ax2.plot(dn_smooth_m*1e6, kana_w, 'k--', label='Kanazawa (Model Baseline)', alpha=0.7)
        self.ax2.plot(exp_dn, exp_dw, 'ro', label='Exp Median')
        self.ax2.set_ylabel(r'$\Delta W$ [ppm]')
        self.ax2.set_xlabel(r'Penetration Depth $\delta$ [$\mu m$]')
        self.ax2.grid(True)
        self.ax2.legend()
        
        self.canvas.draw_idle()

    def apply_viscosity(self):
        new_visc = self.visc_var.get()
        target_entries = self.target_entry if isinstance(self.target_entry, (list, tuple)) else [self.target_entry]
        for entry in target_entries:
            entry.delete(0, 'end')
            entry.insert(0, f"{new_visc:.5f}")
        
        if self.target_btn is not None:
            btn_text = "Auto Calibrated" if self.selection_mode == "auto" else "Manual Calibrated"
            btn_color = "#27AE60" if self.selection_mode == "auto" else "#E67E22"
            hover_color = "#229954" if self.selection_mode == "auto" else "#CA6F1E"
            self.target_btn.configure(text=btn_text, fg_color=btn_color, hover_color=hover_color, width=260)
        if self.group_id is not None:
            self.parent_app.manual_cycle_viscosity_memory[self.group_id] = float(new_visc)
            
        self.parent_app.log(f">>> Cycle Viscosity updated to {new_visc:.5f} Pa·s via Calibration.")
        self.destroy()

# ==========================================
# --- Dynamic Parameters Window ---
# ==========================================

class DynamicParamsWindow(ctk.CTkToplevel):
    def __init__(self, parent_app, run_callback, custom_segments=None, segment_groups=None, window_title=None):
        super().__init__()
        self.title(window_title if window_title else "Dynamic Parameters per Cycle")
        self.geometry("1100x750") 
        self.parent_app = parent_app
        self.run_callback = run_callback
        self.custom_segments = custom_segments
        self.segment_groups = segment_groups
        
        self.after(10, lambda: self.state('zoomed'))
        
        self.cycle_inputs = []

        ctk.CTkLabel(self, text="Define Parameters Per Cycle/Segment", font=("Arial", 16, "bold")).pack(pady=(15, 5))
        ctk.CTkLabel(self, text="Calibrate Electrolyte Viscosity per cycle or enter manually.", 
                     text_color="gray", font=("Arial", 12)).pack(pady=(0, 5))

        top_controls = ctk.CTkFrame(self, fg_color="#2b2b2b", corner_radius=8)
        top_controls.pack(fill="x", padx=20, pady=(10, 10))
        
        self.btn_auto_all = ctk.CTkButton(top_controls, text="Auto Calibrate Viscosity", fg_color="#3498db", hover_color="#2980b9", command=self.auto_calibrate_all, width=220, font=("Arial", 13, "bold"))
        self.btn_auto_all.pack(side="left", padx=15, pady=10)

        self.btn_reset_splits = ctk.CTkButton(
            top_controls,
            text="Reset Manual Splits",
            fg_color="#6b7280",
            hover_color="#4b5563",
            command=self.reset_manual_splits,
            width=180,
            font=("Arial", 13, "bold")
        )
        self.btn_reset_splits.pack(side="left", padx=(0, 15), pady=10)

        ctk.CTkLabel(top_controls, text="Resolution:", font=("Arial", 13)).pack(side="left", padx=(20, 5), pady=10)
        self.res_var = ctk.StringVar(value=self.parent_app.res_var.get())
        self.combo_res = ctk.CTkOptionMenu(top_controls, values=["Low (Fast)", "Normal", "High (Slow)"], variable=self.res_var, width=120)
        self.combo_res.pack(side="left", padx=5, pady=10)

        self.btn_apply_all_visc = ctk.CTkButton(top_controls, text="Apply to All", fg_color="#16a085", hover_color="#1abc9c", command=self.apply_global_viscosity, width=100, font=("Arial", 12, "bold"))
        self.btn_apply_all_visc.pack(side="right", padx=15, pady=10)
        
        # --- Update: Pulling global viscosity value from main app ---
        try:
            current_main_visc = self.parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].get()
        except:
            current_main_visc = "0.00320"
            
        self.entry_global_visc = ctk.CTkEntry(top_controls, width=90, justify="center")
        self.entry_global_visc.insert(0, current_main_visc)
        self.entry_global_visc.pack(side="right", padx=5, pady=10)
        
        ctk.CTkLabel(top_controls, text="Set All Viscosity:", font=("Arial", 13, "bold")).pack(side="right", padx=(20, 5), pady=10)

        self.table_container = ctk.CTkFrame(self, fg_color="transparent")
        self.table_container.pack(fill="both", expand=True, padx=20, pady=5)

        header_frame = ctk.CTkFrame(self.table_container, fg_color="gray20", corner_radius=8)
        header_frame.pack(fill="x", pady=(0, 5), padx=(0, 16))
        
        for i in range(8):
            header_frame.columnconfigure(i, weight=1, uniform="col")
        
        ctk.CTkLabel(header_frame, text="Cycle", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=0, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Segment", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=1, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Time Range (s)", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=2, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Theta (0-1)", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=3, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Viscosity (Pa·s)", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=4, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Calibration", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=5, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Viscosity (Pa·s)", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=3, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Cycle Calibration", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=5, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Cycle Optimization", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=6, padx=5, pady=8, sticky="ew")
        ctk.CTkLabel(header_frame, text="Progress", font=("Arial", 13, "bold"), anchor="center").grid(row=0, column=7, padx=5, pady=8, sticky="ew")

        for child in header_frame.winfo_children():
            child.destroy()

        header_titles = [
            "Cycle",
            "Segment",
            "Time Range (s)",
            "Theta (0-1)",
            "Viscosity (Pa·s)",
            "Viscosity Calibration",
            "Cycle Optimization",
            "Progress"
        ]
        header_weights = [1, 1, 2, 1, 1, 2, 2, 1]
        for i, weight in enumerate(header_weights):
            header_frame.columnconfigure(i, weight=weight, uniform="col")
            ctk.CTkLabel(header_frame, text=header_titles[i], font=("Arial", 13, "bold"), anchor="center").grid(
                row=0, column=i, padx=5, pady=8, sticky="ew"
            )

        self.scroll_frame = ctk.CTkScrollableFrame(self.table_container)
        self.scroll_frame.pack(fill="both", expand=True)

        self.populate_cycles()

        self.bottom_actions = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_actions.pack(pady=20, padx=80, fill="x", side="bottom")

        self.btn_run_multi = ctk.CTkButton(
            self.bottom_actions,
            text="Run Dynamic Optimization (All Cycles)",
            command=self.trigger_run,
            fg_color="green",
            height=45,
            corner_radius=8,
            font=("Arial", 14, "bold")
        )
        self.btn_run_multi.pack(side="left", fill="x", expand=True)

        self.btn_stop_multi = ctk.CTkButton(
            self.bottom_actions,
            text="Stop",
            command=self.request_stop,
            fg_color="#c0392b",
            hover_color="#a93226",
            width=130,
            height=45,
            corner_radius=8,
            font=("Arial", 14, "bold")
        )
        self.btn_stop_multi.pack(side="left", padx=(12, 0))
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def get_time_column_name(self):
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in self.parent_app.working_df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                return col
        return None

    def reset_manual_splits(self):
        self.parent_app.reset_manual_segmentation_state()
        if self.winfo_exists():
            self.destroy()
        self.parent_app.open_dynamic_window()

    def set_calibration_button_state(self, button, state):
        if button is None:
            return

        state_map = {
            "uncalibrated": {"text": "Calibrate", "fg": "#3498db", "hover": "#2980b9"},
            "manual": {"text": "Manual Calibrated", "fg": "#E67E22", "hover": "#CA6F1E"},
            "auto": {"text": "Auto Calibrated", "fg": "#27AE60", "hover": "#229954"},
        }
        cfg = state_map.get(state, state_map["uncalibrated"])
        button.configure(text=cfg["text"], fg_color=cfg["fg"], hover_color=cfg["hover"], width=260)

    def request_stop(self):
        if not self.parent_app.is_running:
            self.parent_app.log(">>> No active fitting process to stop.")
            return
        self.parent_app.log(">>> Stop requested by user...")
        self.parent_app.stop_flag = True

    def update_cycle_viscosity_memory(self, group_id, value):
        try:
            self.parent_app.manual_cycle_viscosity_memory[group_id] = float(value)
        except (TypeError, ValueError):
            pass

    def mark_manual_calibration(self, group_id, visc_entry, btn_calib):
        try:
            current_val = float(visc_entry.get())
        except ValueError:
            return
        self.update_cycle_viscosity_memory(group_id, current_val)
        self.set_calibration_button_state(btn_calib, "manual")

    def populate_cycles(self):
        df = self.parent_app.working_df
        time_col = self.get_time_column_name()
        
        if df is None or not time_col:
            ctk.CTkLabel(self.scroll_frame, text="No valid data or time column found.", text_color="red").pack(pady=20)
            return

        times = pd.to_numeric(df[time_col], errors='coerce').fillna(0).to_numpy()
        
        try:
            global_theta = float(self.parent_app.entries["Coverage (θ) [range: 0–1]"].get())
            global_visc = float(self.parent_app.entries["Ref. Liquid Viscosity [Pa·s]"].get())
        except:
            global_theta = 1.0
            global_visc = 0.0032

        if self.segment_groups is not None:
            grouped_segments = self.segment_groups
        else:
            cycles = self.parent_app.cycle_indices
            if cycles is None or len(cycles) < 2:
                ctk.CTkLabel(self.scroll_frame, text="No cycles detected. Please ensure a successful full run.", text_color="red").pack(pady=20)
                return
            grouped_segments = []
            for i in range(0, len(cycles) - 2, 2):
                grouped_segments.append({'label': f"#{(i // 2) + 1}", 'segments': [(int(cycles[i]), int(cycles[i+2]))]})

        for cycle_count, group in enumerate(grouped_segments, start=1):
            group_segments = sorted(group['segments'], key=lambda seg: seg[0])
            group_start = group_segments[0][0]
            group_end = group_segments[-1][1]
            row_count = len(group_segments)
            cycle_default_visc = self.parent_app.manual_cycle_viscosity_memory.get(cycle_count)
            if cycle_default_visc is None and "Used_Viscosity_Pa_s" in df.columns:
                cycle_visc_series = pd.to_numeric(df.loc[group_start:group_end, "Used_Viscosity_Pa_s"], errors="coerce").dropna()
                if not cycle_visc_series.empty:
                    cycle_default_visc = float(cycle_visc_series.iloc[-1])
                    self.parent_app.manual_cycle_viscosity_memory[cycle_count] = cycle_default_visc
            if cycle_default_visc is None:
                cycle_default_visc = global_visc

            group_frame = ctk.CTkFrame(
                self.scroll_frame,
                fg_color="#2f2f2f" if row_count > 1 else "transparent",
                corner_radius=8 if row_count > 1 else 0
            )
            group_frame.pack(fill="x", pady=4, padx=2)

            column_weights = [1, 1, 2, 1, 1, 2, 2, 1]
            for col_idx, col_weight in enumerate(column_weights):
                group_frame.columnconfigure(col_idx, weight=col_weight, uniform="col")
            for row_idx in range(row_count):
                group_frame.rowconfigure(row_idx, weight=1)

            cycle_label = group.get('label', f"#{cycle_count}")
            cycle_cell = ctk.CTkFrame(group_frame, fg_color="transparent")
            cycle_cell.grid(row=0, column=0, rowspan=row_count, padx=5, pady=6, sticky="nsew")
            ctk.CTkLabel(cycle_cell, text=cycle_label, anchor="center").pack(expand=True)

            optimize_cell = ctk.CTkFrame(group_frame, fg_color="transparent")
            optimize_cell.grid(row=0, column=6, rowspan=row_count, padx=12, pady=6, sticky="nsew")
            progress_cell = ctk.CTkFrame(group_frame, fg_color="transparent")
            progress_cell.grid(row=0, column=7, rowspan=row_count, padx=5, pady=6, sticky="nsew")

            cycle_progress_label = ctk.CTkLabel(progress_cell, text="", text_color="gray", anchor="center")
            btn_opt_single = ctk.CTkButton(
                optimize_cell,
                text=f"Optimize Cycle {cycle_count}",
                fg_color="#8E44AD",
                hover_color="#732D91",
                command=lambda grp_id=cycle_count, p_lbl=cycle_progress_label: self.run_cycle_group(grp_id, p_lbl)
            )
            btn_opt_single.pack(expand=True, fill="x")
            cycle_progress_label.pack(expand=True, fill="x")

            for seg_idx, (idx_start, idx_end) in enumerate(group_segments, start=1):
                row_idx = seg_idx - 1
                t_start = times[idx_start]
                t_end = times[idx_end]
                segment_label = f"S{seg_idx}" if row_count > 1 else "Main"

                ctk.CTkLabel(group_frame, text=segment_label, anchor="center").grid(row=row_idx, column=1, padx=5, pady=6, sticky="ew")
                ctk.CTkLabel(group_frame, text=f"{t_start:.1f} - {t_end:.1f}", anchor="center").grid(row=row_idx, column=2, padx=5, pady=6, sticky="ew")

                entry_theta = ctk.CTkEntry(group_frame, justify="center")
                entry_theta.insert(0, str(global_theta))
                entry_theta.grid(row=row_idx, column=3, padx=12, pady=6, sticky="ew")

                entry_visc = ctk.CTkEntry(group_frame, justify="center")
                entry_visc.insert(0, f"{cycle_default_visc:.5f}")
                entry_visc.grid(row=row_idx, column=4, padx=12, pady=6, sticky="ew")

                btn_calib = ctk.CTkButton(group_frame, text="Calibrate", fg_color="#3498db", hover_color="#2980b9", width=260)
                btn_calib.configure(command=lambda s_idx=idx_start, e_idx=idx_end, ent=entry_visc, btn=btn_calib, grp=cycle_count: self.open_cycle_calibration(s_idx, e_idx, ent, btn, grp))
                btn_calib.grid(row=row_idx, column=5, padx=12, pady=6, sticky="ew")
                entry_visc.bind("<KeyRelease>", lambda e, grp=cycle_count, ent=entry_visc, btn=btn_calib: self.mark_manual_calibration(grp, ent, btn))
                entry_visc.bind("<FocusOut>", lambda e, grp=cycle_count, ent=entry_visc, btn=btn_calib: self.mark_manual_calibration(grp, ent, btn))

                self.cycle_inputs.append({
                    'start_time': t_start,
                    'end_time': t_end,
                    'start_idx': idx_start,
                    'end_idx': idx_end,
                    'theta_entry': entry_theta,
                    'visc_entry': entry_visc,
                    'btn_calib': btn_calib,
                    'group_id': cycle_count,
                    'group_start_idx': group_start,
                    'group_end_idx': group_end
                })

    def apply_global_viscosity(self):
        try:
            val = float(self.entry_global_visc.get())
            if val <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Error", "Invalid global viscosity value. Must be a positive number.", parent=self)
            return
        
        val_str = f"{val:.5f}"
        for inp in self.cycle_inputs:
            inp['visc_entry'].delete(0, 'end')
            inp['visc_entry'].insert(0, val_str)
            self.update_cycle_viscosity_memory(inp['group_id'], val)
            if 'btn_calib' in inp and inp['btn_calib'] is not None:
                self.set_calibration_button_state(inp['btn_calib'], "manual")
                
        self.parent_app.log(f">>> Global Viscosity applied to all cycles: {val_str} Pa·s.")

    def run_cycle_group(self, group_id, progress_label):
        if self.parent_app.is_running:
            self.parent_app.log(">>> Stopping current process to restart cycle optimization...")
            self.parent_app.stop_flag = True
            progress_label.configure(text="Stopping...", text_color="red")
            self.update()
            self.after(300, lambda: self.run_cycle_group(group_id, progress_label))
            return

        group_rows = [inp for inp in self.cycle_inputs if inp['group_id'] == group_id]
        if not group_rows:
            messagebox.showerror("Error", "No rows found for this cycle.", parent=self)
            return

        try:
            dynamic_visc = []
            dynamic_theta = []
            for inp in group_rows:
                v_val = float(inp['visc_entry'].get())
                t_val = float(inp['theta_entry'].get())
                if v_val <= 0 or t_val < 0 or t_val > 1:
                    raise ValueError
                dynamic_visc.append({'start': inp['start_time'], 'end': inp['end_time'], 'val': v_val})
                dynamic_theta.append({'start': inp['start_time'], 'end': inp['end_time'], 'val': t_val})
        except ValueError:
            messagebox.showerror("Error", "Invalid Theta or Viscosity values.", parent=self)
            return

        start_idx = min(inp['start_idx'] for inp in group_rows)
        end_idx = max(inp['end_idx'] for inp in group_rows)

        progress_label.configure(text="Starting...", text_color="#FF8C00")
        self.update()

        p = self.parent_app.get_params()
        if not p: return
        
        res_choice = self.combo_res.get()
        if "High" in res_choice: calc_tol, calc_pop = 0.001, 40
        elif "Low" in res_choice: calc_tol, calc_pop = 0.1, 10
        else: calc_tol, calc_pop = 0.01, 20
        
        df_slice = self.parent_app.working_df.loc[start_idx : end_idx].copy()
        dynamic_config = {'visc': dynamic_visc, 'theta': dynamic_theta}
        
        def prog_cb(current, total):
            progress_label.configure(text=f"{current}/{total}")
            self.update()
            
        self.parent_app.is_running = True
        self.parent_app.stop_flag = False
        
        res_h, res_xi, res_cutoff, res_status, res_curves, res_exp_3rd, res_visc, res_theta, succ_pts = \
            self.parent_app.calculate_optimization_loop(p, calc_tol, calc_pop, dynamic_config=dynamic_config, df_subset=df_slice, progress_callback=prog_cb)
            
        self.parent_app.is_running = False
        
        if res_h is None:
            progress_label.configure(text="Stopped", text_color="red")
            return
            
        indices = df_slice.index
        self.parent_app.working_df.loc[indices, "Optimized_Height_nm"] = res_h
        self.parent_app.working_df.loc[indices, "Optimized_Qsi_nm"] = res_xi
        self.parent_app.working_df.loc[indices, "Cutoff_nm"] = res_cutoff
        self.parent_app.working_df.loc[indices, "Fit_Status"] = res_status
        self.parent_app.working_df.loc[indices, "Used_Viscosity_Pa_s"] = res_visc
        self.parent_app.working_df.loc[indices, "Used_Theta"] = res_theta
        
        self.parent_app.working_df["Graph_h_nm"] = self.parent_app.working_df["Optimized_Height_nm"].interpolate(method='linear')
        self.parent_app.working_df["Graph_xi_nm"] = self.parent_app.working_df["Optimized_Qsi_nm"].interpolate(method='linear')
        
        progress_label.configure(text="Done!", text_color="green")
        
        df_cycle_view = self.parent_app.working_df.loc[start_idx : end_idx].copy()
        fname = f"{self.parent_app.filename_qcmd}_Cycle_{group_id}"
        
        self.parent_app.open_combined_graph_window(cycle_times=None, cycle_indices=None, filename_override=fname, df_override=df_cycle_view)

    def open_cycle_calibration(self, start_idx, end_idx, target_entry, target_btn, group_id=None):
        df_slice = self.parent_app.working_df.loc[start_idx:end_idx].copy()
        
        def on_confirm(selected_idx):
            CycleViscosityCalibrationWindow(self.parent_app, start_idx, selected_idx, target_entry, target_btn, group_id=group_id)
            
        highest_n = max(self.parent_app.current_harmonics)
        col_d_highest = f"D{highest_n}" if f"D{highest_n}" in df_slice.columns else (f"d{highest_n}" if f"d{highest_n}" in df_slice.columns else None)
        
        if col_d_highest and not df_slice.empty:
            default_idx = df_slice[col_d_highest].idxmin()
        else:
            default_loc = 5 if len(df_slice) > 5 else (len(df_slice)-1 if len(df_slice)>0 else 0)
            default_idx = df_slice.index[default_loc]
        
        bs_win = BaselineSelectionWindow(self.parent_app, df_slice, self.parent_app.current_harmonics, on_confirm, default_idx)
        bs_win.lift()

    def auto_calibrate_all(self):
        p = self.parent_app.get_params()
        if not p: return
        
        df = self.parent_app.working_df
        harmonics = self.parent_app.current_harmonics
        
        weights_list = [10 if k < 3 else 1.0 for k in range(len(harmonics))]
        f_weights_arr = np.array(weights_list)

        self.btn_auto_all.configure(state="disabled", text="Calibrating...")
        self.update_idletasks()

        for inp in self.cycle_inputs:
            start_idx = inp['start_idx']
            end_idx = inp['end_idx']
            
            highest_n = max(harmonics)
            col_d_highest = f"D{highest_n}" if f"D{highest_n}" in df.columns else (f"d{highest_n}" if f"d{highest_n}" in df.columns else None)
            
            if col_d_highest and start_idx in df.index and end_idx in df.index:
                cycle_slice = df.loc[start_idx:end_idx]
                if not cycle_slice.empty:
                    target_idx = cycle_slice[col_d_highest].idxmin()
                else:
                    target_idx = start_idx
            else:
                target_idx = start_idx + 5
                if len(df) <= target_idx:
                    target_idx = len(df) - 1
                    if target_idx < start_idx: target_idx = start_idx
            
            target_df = []
            target_dw = []
            
            valid = True
            for n in harmonics:
                col_f = f"f{n}"
                col_d = f"D{n}" if f"D{n}" in df.columns else f"d{n}"
                
                try:
                    med_f = float(df.loc[target_idx, col_f])
                    med_d = float(df.loc[target_idx, col_d])
                except:
                    valid = False
                    break
                    
                f_air = p["air_f"][n]
                d_air = p["air_d"][n]
                
                delta_f = med_f - f_air
                denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
                num_f = (delta_f / n) - 0 
                calc_df = num_f / denominator
                
                term_w_liq = (med_d * 1e-6) * (med_f / n)
                term_w_air = (d_air * 1e-6) * (f_air / n)
                num_w = term_w_liq - term_w_air
                calc_dw = num_w / denominator
                
                target_df.append(calc_df * 1e9)
                target_dw.append(calc_dw * 1e9)
                
            if not valid: continue
            
            target_df = np.array(target_df)
            target_dw = np.array(target_dw)

            def objective_function(params_opt):
                visc_cand = params_opt[0]
                row_dn_um = []
                for n in harmonics:
                    f_air = p["air_f"][n]
                    val_dn = 1e6 * np.sqrt(visc_cand / (np.pi * p["rho_liq"] * f_air))
                    row_dn_um.append(val_dn)
                row_dn_meters = np.array(row_dn_um) * 1e-6
                
                w_kana, f_kana = model_Kanazawa_line(row_dn_meters, p["rho_quartz"], p["visc_quartz"])
                if w_kana is None: return 1e9
                
                error_w = np.sum((w_kana - target_dw)**2)
                error_f = np.sum(((f_kana - target_df)**2) * f_weights_arr)
                return (100 * error_w) + error_f

            bounds = [(0.0001, 0.1)] 
            result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=500, popsize=10, tol=1e-3)
            
            if result.success:
                best_visc = result.x[0]
                inp['visc_entry'].delete(0, 'end')
                inp['visc_entry'].insert(0, f"{best_visc:.5f}")
                self.update_cycle_viscosity_memory(inp['group_id'], best_visc)
                
                if 'btn_calib' in inp and inp['btn_calib'] is not None:
                    self.set_calibration_button_state(inp['btn_calib'], "auto")
            
            self.update() 

        self.btn_auto_all.configure(state="normal", text="Auto Calibrate Viscosity")
        messagebox.showinfo("Done", "Auto calibration for eligible cycles completed.", parent=self)

    def trigger_run(self):
        self.parent_app.res_var.set(self.res_var.get())
        
        if not self.cycle_inputs:
            messagebox.showerror("Error", "No cycles to optimize.", parent=self)
            return

        visc_ranges = []
        theta_ranges = []
        
        for inp in self.cycle_inputs:
            try:
                v = float(inp['visc_entry'].get())
                t = float(inp['theta_entry'].get())
                if v <= 0 or t < 0 or t > 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "All entries must be filled with valid numbers (Theta 0-1, Visc > 0).", parent=self)
                return
                
            visc_ranges.append({'start': inp['start_time'], 'end': inp['end_time'], 'val': v})
            theta_ranges.append({'start': inp['start_time'], 'end': inp['end_time'], 'val': t})
            
        config = {
            'visc': visc_ranges,
            'theta': theta_ranges
        }
        self.destroy()
        self.parent_app.after(200, lambda: self.run_callback(config))


# === COMBINED ANALYSIS WINDOW ===
class CombinedGraphWindow(ctk.CTkToplevel):
    def __init__(self, df, harmonics, filename_base, on_click_callback, open_manual_callback=None, cycle_times=None, cycle_indices=None, mode_label="h", filename_override=None, global_f0=None, global_d0=None):
        super().__init__()
        
        title_text = f"Combined Analysis: F, D, h, Xi | {filename_override}" if filename_override else "Combined Analysis: F, D, h, Xi"
        self.title(title_text)
        
        self.geometry("1300x950") 
        self.after(10, lambda: self.state('zoomed'))
        
        self.df = df
        self.filename_base = filename_override if filename_override else filename_base
        self.on_click_callback = on_click_callback
        self.open_manual_callback = open_manual_callback
        self.cycle_times = cycle_times
        self.cycle_indices = cycle_indices
        self.mode_label = mode_label
        
        self.global_f0 = global_f0 if global_f0 is not None else {}
        self.global_d0 = global_d0 if global_d0 is not None else {}
        
        self.x_data = None
        self.x_label = "Index"
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                self.x_data = pd.to_numeric(df[col], errors='coerce').fillna(0).to_numpy()
                self.x_label = col
                break
        if self.x_data is None: self.x_data = df.index.to_numpy()

        self.y_height_abs = pd.to_numeric(df["Graph_h_nm"], errors='coerce').to_numpy()
        self.y_qsi = pd.to_numeric(df["Graph_xi_nm"], errors='coerce').to_numpy()
        
        self.y_theo = None
        if "Theo_Calibrated_Active" in df.columns:
            self.y_theo = pd.to_numeric(df["Theo_Calibrated_Active"], errors='coerce').to_numpy()
            
        self.y_voltage = None
        if "E_we_V" in df.columns:
            self.y_voltage = pd.to_numeric(df["E_we_V"], errors='coerce').to_numpy()

        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=10, fill="x")

        if self.open_manual_callback is not None:
            btn_manual = ctk.CTkButton(
                btn_frame,
                text="Manual Segmentation",
                command=self.open_manual_callback,
                fg_color="#9B59B6",
                hover_color="#8E44AD",
                width=160
            )
            btn_manual.pack(side="left", padx=40)
        
        btn_save_xls = ctk.CTkButton(btn_frame, text="Combined Analysis (Excel)", command=self.save_xls, fg_color="green", width=120)
        btn_save_xls.pack(side="right", padx=(10, 40))
        
        btn_save_img = ctk.CTkButton(btn_frame, text="Combined Analysis (IMAGE)", command=self.save_img, fg_color="#1f538d", width=120)
        btn_save_img.pack(side="right", padx=(10, 10))

        self.h_view_var = ctk.StringVar(value="Absolute h")
        self.seg_h_view = ctk.CTkSegmentedButton(btn_frame, values=["Absolute h", "Relative h"],
                                                 command=self.update_h_plot, variable=self.h_view_var)
        self.seg_h_view.place(relx=0.5, rely=0.5, anchor="center")

        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(10, 9), sharex=False)
        self.fig.suptitle("Double-click any point in the h or ξ plot to view the corresponding fitting result.", fontsize=11, color='#333333', fontweight='bold')
        self.fig.subplots_adjust(right=0.85, hspace=0.3, top=0.92, bottom=0.05, left=0.1) 
        
        self.legend_pos = (1.05, 1)
        self.harmonics = harmonics

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        self.plot_static_graphs()
        self.update_h_plot("Absolute h")

        self.canvas.mpl_connect('button_press_event', self.on_plot_click)
        
        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def plot_static_graphs(self):
        f_colors_fixed = {3: "#0819CE", 5: "#1143E5", 7: "#2E71E9", 9: "#68A9E8", 11: "#6DBCE4", 13: "#8CD3E6"}
        d_colors_fixed = {3: "#FF2C14", 5: "#FD5328", 7: "#FD753A", 9: "#FAA56D", 11: "#FBB261", 13: "#FBC873"}

        self.ax1.set_ylabel(r"$\Delta f_n / n$ [Hz]", fontsize=9)
        if self.y_voltage is not None:
            self.ax1_twin = self.ax1.twinx()
            self.ax1_twin.set_ylabel("E [V]", color='#333333', fontsize=9)
            self.ax1_twin.plot(self.x_data, self.y_voltage, color='#333333', alpha=0.9, linewidth=1.2, linestyle='-', label='E [V]')
            self.ax1_twin.tick_params(axis='y', labelcolor='#333333')

        for n in self.harmonics:
            col_name = f"f{n}"
            if col_name in self.df.columns:
                raw_f = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0).to_numpy()
                f0 = self.global_f0.get(n, raw_f[0] if len(raw_f) > 0 else 0)
                norm_f = (raw_f - f0) / n
                c = f_colors_fixed.get(n, plt.cm.Blues(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax1.plot(self.x_data, norm_f, color=c, linewidth=1.0, alpha=0.7, label=f"f{n}")
                
        if self.y_theo is not None:
            self.ax1.plot(self.x_data, self.y_theo, color='black', linestyle='-.', linewidth=1.5, label=r'$\Delta f_{theo} (Calib)$')
            
        lines, labels = self.ax1.get_legend_handles_labels()
        if self.y_voltage is not None:
            lines2, labels2 = self.ax1_twin.get_legend_handles_labels()
            lines += lines2; labels += labels2
        self.ax1.legend(lines, labels, bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax1.grid(True, linestyle='--', alpha=0.5)

        self.ax2.set_ylabel(r"$\Delta D_n$ [ppm]", fontsize=9)
        for n in self.harmonics:
            col_name = f"D{n}"
            if col_name not in self.df.columns: col_name = f"d{n}"
            if col_name in self.df.columns:
                raw_d = pd.to_numeric(self.df[col_name], errors='coerce').fillna(0).to_numpy()
                d0 = self.global_d0.get(n, raw_d[0] if len(raw_d) > 0 else 0)
                delta_d = (raw_d - d0) 
                c = d_colors_fixed.get(n, plt.cm.YlOrRd(np.clip(0.45 - (n - 13) * 0.03, 0.1, 1.0)))
                self.ax2.plot(self.x_data, delta_d, color=c, linewidth=1.2, label=rf"$\Delta D_{{{n}}}$")
        self.ax2.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax2.grid(True, linestyle='--', alpha=0.5)

        self.ax4.set_ylabel(r"$\xi$ [nm]", fontsize=9)
        self.ax4.plot(self.x_data, self.y_qsi, color='#ff7f0e', linewidth=1.5, label=r'Permeability length ($\xi$)')
        self.ax4.set_xlabel(self.x_label)
        self.ax4.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.ax4.grid(True, linestyle='--', alpha=0.5)

        if self.cycle_times is not None:
            for ct in self.cycle_times:
                self.ax1.axvline(x=ct, color='gray', linestyle='--', linewidth=1.0, alpha=0.5)

    def calculate_relative_h(self):
        if self.cycle_indices is None or len(self.cycle_indices) < 3:
            return self.y_height_abs

        h_rel = np.copy(self.y_height_abs)
        for i in range(0, len(self.cycle_indices) - 2, 2):
            idx_start = self.cycle_indices[i]
            idx_end = self.cycle_indices[i+2]
            
            cycle_slice = h_rel[idx_start : idx_end + 1]
            if len(cycle_slice) > 0:
                valid_vals = [v for v in cycle_slice if v is not None and not np.isnan(v)]
                if valid_vals:
                    min_val = np.min(valid_vals)
                    new_slice = []
                    for v in cycle_slice:
                        if v is not None and not np.isnan(v):
                            new_slice.append(v - min_val)
                        else:
                            new_slice.append(v)
                    h_rel[idx_start : idx_end + 1] = new_slice
        return h_rel

    def update_h_plot(self, mode):
        self.ax3.clear()
        self.ax3.set_ylabel(r"$h$ [nm]", fontsize=9)
        self.ax3.grid(True, linestyle='--', alpha=0.5)
        
        if mode == "Absolute h":
            y_data = self.y_height_abs
            label_txt = r'Total Height ($h_{total}$)'
            color = '#1f77b4'
        else: # mode == "Relative h"
            y_data = self.calculate_relative_h()
            label_txt = r'Relative Height ($h_{rel}$)'
            color = '#2ca02c' 

        self.ax3.plot(self.x_data, y_data, color=color, linewidth=1.5, label=label_txt)
        
        if self.cycle_times is not None:
            for ct in self.cycle_times:
                self.ax3.axvline(x=ct, color='gray', linestyle='--', linewidth=1.5, alpha=0.8)
                
        self.ax3.legend(bbox_to_anchor=self.legend_pos, loc='upper left', borderaxespad=0, fontsize='small')
        self.canvas.draw()

    def on_plot_click(self, event):
        if not event.dblclick: return
        if self.toolbar.mode != "": return
        is_ax1 = (event.inaxes == self.ax1)
        if hasattr(self, 'ax1_twin'):
            is_ax1 = is_ax1 or (event.inaxes == self.ax1_twin)
            
        if (is_ax1 or event.inaxes in [self.ax2, self.ax3, self.ax4]) and event.xdata is not None:
            distances = np.abs(self.x_data - event.xdata)
            nearest_idx = distances.argmin()
            actual_idx = self.df.index[nearest_idx]
            self.on_click_callback(actual_idx)

    def save_img(self):
        current_mode = self.h_view_var.get()
        mode_suffix = current_mode.replace(" ", "_")
        
        default_name = f"{self.filename_base}_Combined_Analysis_{mode_suffix}"
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            initialfile=default_name,
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path)
            messagebox.showinfo("Saved", "Graph saved successfully.")

    def save_xls(self):
        current_mode = self.h_view_var.get()
        mode_suffix = "Absolute"
        if "Relative" in current_mode: mode_suffix = "Relative"
        
        export_df = self.df.copy()
        
        invalid_statuses = ["Failed", "Failed_Validation", "Error", "Missing Cols"]
        export_df = export_df[~export_df["Fit_Status"].isin(invalid_statuses)]
        
        abs_data = export_df["Optimized_Height_nm"].to_numpy()
        
        if mode_suffix == "Relative":
            h_rel = np.copy(abs_data)
            if self.cycle_indices is not None and len(self.cycle_indices) >= 3:
                for i in range(0, len(self.cycle_indices) - 2, 2):
                    idx_start = self.cycle_indices[i]
                    idx_end = self.cycle_indices[i+2]
                    cycle_slice = export_df.loc[(export_df.index >= idx_start) & (export_df.index <= idx_end), "Optimized_Height_nm"]
                    if len(cycle_slice) > 0:
                        min_val = cycle_slice.min()
                        export_df.loc[cycle_slice.index, "Optimized_Height_nm"] = cycle_slice - min_val
            
        col_name = f"Optimized_Height_nm_{mode_suffix}"
        export_df.rename(columns={"Optimized_Height_nm": col_name}, inplace=True)
        
        cols_to_drop = ["Graph_h_nm", "Graph_xi_nm", "F_Calibrated_View"]
        for c in cols_to_drop:
            if c in export_df.columns:
                export_df.drop(columns=[c], inplace=True)

        default_name = f"{self.filename_base}_Final_Results_{mode_suffix}_h"
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                            initialfile=default_name,
                                            filetypes=[("Excel", "*.xlsx")])
        if path:
            export_df.to_excel(path, index=False)
            messagebox.showinfo("Saved", f"Data saved as {mode_suffix} h successfully.\nNote: Invalid/Failed points were automatically removed.")

# === ROW WINDOW CLASS ===
class RowGraphWindow(ctk.CTkToplevel):
    def __init__(self, idx, time_val, best_h, best_xi, 
                 dn_model, f_model, w_model, 
                 dn_exp, f_exp, w_exp,
                 kana_f, kana_w,
                 filename_base, export_data_dict, open_manual_callback=None):
        super().__init__()
        self.title(f"Row {idx} Analysis")
        self.geometry("900x800")
        self.after(10, lambda: self.state('zoomed'))
        
        self.filename_base = filename_base
        self.idx = idx
        self.export_data_dict = export_data_dict
        self.open_manual_callback = open_manual_callback

        main_container = ctk.CTkFrame(self, fg_color="transparent")
        main_container.pack(fill="both", expand=True, pady=(0, 10))

        toolbar_frame = ctk.CTkFrame(main_container, width=45, corner_radius=0, fg_color="#404040")
        toolbar_frame.pack(side="left", fill="y")

        graph_frame = ctk.CTkFrame(main_container, fg_color="transparent")
        graph_frame.pack(side="right", fill="both", expand=True)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(side="bottom", pady=10, fill="x")

        if self.open_manual_callback is not None:
            btn_manual = ctk.CTkButton(
                btn_frame,
                text="Manual Segmentation",
                command=self.open_manual_callback,
                fg_color="#9B59B6",
                hover_color="#8E44AD"
            )
            btn_manual.pack(side="left", padx=20)
        
        btn_save_xls = ctk.CTkButton(btn_frame, text=f"Row {idx} Analysis (Excel)", command=self.save_xls, fg_color="green")
        btn_save_xls.pack(side="right", padx=(10, 20))

        btn_save_img = ctk.CTkButton(btn_frame, text=f"Row {idx} Analysis (IMAGE)", command=self.save_img, fg_color="#1f538d")
        btn_save_img.pack(side="right", padx=(10, 10))
        
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(6, 8))
        plt.subplots_adjust(bottom=0.15, left=0.15, right=0.95)
        
        self.fig.suptitle(fr"Row {idx} | Time: {time_val}sec" + "\n" + r"$h$=" + f"{best_h:.1f}, " + r"$\xi$=" + f"{best_xi:.1f}")
        
        self.ax1.plot(dn_model*1e6, f_model, 'b-', label='Model')
        if kana_f is not None: self.ax1.plot(dn_model*1e6, kana_f, 'k--', label='Kanazawa', alpha=0.7)
        self.ax1.plot(dn_exp, f_exp, 'bo', label=r'Exp')
        self.ax1.set_ylabel(r'$\Delta f / n$ [Hz]'); self.ax1.grid(True); self.ax1.legend()
        
        self.ax2.plot(dn_model*1e6, w_model, 'r-', label='Model')
        if kana_w is not None: self.ax2.plot(dn_model*1e6, kana_w, 'k--', label='Kanazawa', alpha=0.7)
        self.ax2.plot(dn_exp, w_exp, 'ro', label='Exp')
        self.ax2.set_ylabel(r'$\Delta W$ [ppm]'); self.ax2.set_xlabel(r'Penetration Depth $\delta$ [$\mu m$]')
        self.ax2.grid(True); self.ax2.legend()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        
        self.toolbar = CustomVerticalToolbar(self.canvas, toolbar_frame)
        self.toolbar.update()
        self.toolbar.pack(side="top", fill="y") 

        self.lift()
        self.attributes('-topmost', True)
        self.after(10, lambda: self.attributes('-topmost', False))
        self.focus_force()

    def save_img(self):
        default_name = f"{self.filename_base}_AUTOFIT_Row_{self.idx}_Graph"
        path = filedialog.asksaveasfilename(defaultextension=".png", 
                                            initialfile=default_name,
                                            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf")])
        if path:
            self.fig.savefig(path)
            messagebox.showinfo("Saved", "Row Graph saved successfully.")

    def save_xls(self):
        df_export = pd.DataFrame(self.export_data_dict)
        default_name = f"{self.filename_base}_AUTOFIT_Row_{self.idx}_Data"
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", 
                                            initialfile=default_name,
                                            filetypes=[("Excel", "*.xlsx")])
        if path:
            df_export.to_excel(path, index=False)
            messagebox.showinfo("Saved", "Row Data saved successfully.")

# ==========================================
# --- 3. GUI Application Class ---
# ==========================================

class PhysicsOptimizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("QCM Analysis & Optimization Tool")
        self.geometry("1200x950")
        self.after(10, lambda: self.state('zoomed'))
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # --- Data Containers ---
        self.df = None          
        self.df_echem = None    
        self.df_original = None         
        self.df_echem_original = None   
        self.current_time_offset = 0.0  
        
        self.raw_air_data = {'f': {}, 'd': {}}
        self.raw_material_data = {'f': {}, 'd': {}}
        self.raw_coated_material_data = {'f': {}, 'd': {}}
        
        self.baseline_indices = None 
        self.cycle_indices = None 
        
        # --- NEW: Memory for Manual Segmentation ---
        self.saved_manual_splits = []
        self.manual_split_memory = {}
        self.manual_cycle_segment_groups = {}
        self.manual_cycle_viscosity_memory = {}
        
        self.filename_qcmd = "Results" 
        self.filename_echem = "EChem_Data"
        
        self.current_harmonics = [3, 5, 7, 9, 11] 
        self.harmonics_widgets = [] 
        self.entries = {}
        self.air_f_entries = {}
        self.air_d_entries = {}
        
        self.is_running = False
        self.stop_flag = False
        self.full_run_completed = False 
        
        self.harmonics_frame = None

        self.combined_window = None 
        self.row_window = None 
        self.dynamic_window = None 
        self.sync_window = None
        self.visc_calib_window = None

        self.setup_ui()
        self.create_inputs_frame()
        self.update_harmonic_fields()

    def truncate_string(self, text, max_len=35):
        if len(text) > max_len:
            return text[:max_len-3] + "..."
        return text

    def setup_ui(self):
        self.step1_frame = ctk.CTkFrame(self)
        self.step1_frame.pack(pady=10, padx=20, fill="x")
        
        self.step1_frame.columnconfigure(0, weight=1, uniform="cols")
        self.step1_frame.columnconfigure(1, weight=1, uniform="cols")
        self.step1_frame.columnconfigure(2, weight=1, uniform="cols")
        
        load_btn_color = "#7C3AED" 
        load_btn_hover = "#6D28D9" 

        self.btn_load_qcmd = ctk.CTkButton(self.step1_frame, text="1. Load QCM-D Data (Excel)", 
                                           command=self.load_qcmd_file,
                                           fg_color=load_btn_color, hover_color=load_btn_hover)
        self.btn_load_qcmd.grid(row=0, column=0, padx=10, pady=(10, 5))
        self.lbl_file_qcmd = ctk.CTkLabel(self.step1_frame, text="No file selected", text_color="gray", width=200)
        self.lbl_file_qcmd.grid(row=1, column=0, padx=10, pady=(0, 10))

        self.btn_load_echem = ctk.CTkButton(self.step1_frame, text="2. Load EChem Data (Excel)", 
                                            command=self.load_echem_file, 
                                            fg_color=load_btn_color, hover_color=load_btn_hover)
        self.btn_load_echem.grid(row=0, column=1, padx=10, pady=(10, 5))
        self.lbl_file_echem = ctk.CTkLabel(self.step1_frame, text="No file selected", text_color="gray", width=200)
        self.lbl_file_echem.grid(row=1, column=1, padx=10, pady=(0, 10))

        self.btn_load_air = ctk.CTkButton(self.step1_frame, text="3. Load Quartz in Air (Excel)", 
                                            command=self.load_air_file, 
                                            fg_color=load_btn_color, hover_color=load_btn_hover)
        self.btn_load_air.grid(row=0, column=2, padx=10, pady=(10, 5))
        self.lbl_file_air = ctk.CTkLabel(self.step1_frame, text="No file selected", text_color="gray", width=200)
        self.lbl_file_air.grid(row=1, column=2, padx=10, pady=(0, 10))
        self.lbl_file_material = None
        self.lbl_file_coated_material = None

        self.scroll_frame = ctk.CTkScrollableFrame(self, height=450)
        self.scroll_frame.pack(pady=10, padx=20, fill="both", expand=True)

        self.action_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.action_frame.pack(pady=20, fill="x")
        
        self.res_var = ctk.StringVar(value="Normal")
        
        center_buttons_frame = ctk.CTkFrame(self.action_frame, fg_color="transparent")
        center_buttons_frame.pack(anchor="center")
        
        self.btn_sync = ctk.CTkButton(center_buttons_frame, text="Time Sync", command=self.open_sync_window, fg_color="#F59E0B", hover_color="#D97706", width=120, height=40, font=("Arial", 12, "bold"))
        self.btn_sync.pack(side="left", padx=10)

        self.btn_main_hub = ctk.CTkButton(center_buttons_frame, text="Auto-Cycles Optimization", command=self.open_dynamic_window, fg_color="#2ECC71", hover_color="#27AE60", width=260, height=40, font=("Arial", 12, "bold"))
        self.btn_main_hub.pack(side="left", padx=10)
        
        self.btn_quick_guide = ctk.CTkButton(center_buttons_frame, text="Quick Guide", command=self.show_quick_guide, fg_color="#4B5563", hover_color="#374151", width=120, height=40, font=("Arial", 12, "bold"))
        self.btn_quick_guide.pack(side="left", padx=10)

        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.pack(pady=(0, 5), fill="x", padx=40)
        
        self.lbl_sync_ref = ctk.CTkLabel(self.progress_frame, text="", font=("Arial", 11), height=10, text_color="gray")
        self.lbl_sync_ref.pack(side="top")
        
        self.lbl_progress_text = ctk.CTkLabel(self.progress_frame, text="Waiting to start... (0/0)", font=("Arial", 12))
        self.lbl_progress_text.pack(pady=5)

    def show_quick_guide(self):
        if hasattr(self, 'guide_window') and self.guide_window is not None and self.guide_window.winfo_exists():
            self.guide_window.focus()
            return

        self.guide_window = ctk.CTkToplevel(self)
        self.guide_window.title("Quick Guide")
        self.guide_window.geometry("780x470") 
        self.guide_window.lift()
        self.guide_window.attributes('-topmost', True)
        self.guide_window.after(10, lambda: self.guide_window.attributes('-topmost', False))

        ctk.CTkLabel(self.guide_window, text="Quick Start Guide", font=("Arial", 16, "bold"), text_color="#00BFFF").pack(pady=(15, 10))
        
        instructions_text = (
            "1. Load the QCM-D, EChem, and Quartz in Air data.\n\n"
            "2. Verify the QCM physical and electrochemical parameters.\n\n"
            "3. Use Time Sync to align the datasets.\n\n"
            "4. Open Auto-Cycles Optimization to detect cycles and define cycle-specific settings.\n\n"
            "5. Set All Viscosity manually and click Apply to All, or use Auto Calibrate Viscosity.\n\n"
            "6. Adjust Theta if the quartz crystal coverage changes.\n\n"
            "7. Use Viscosity Calibration to calibrate the starting viscosity of each cycle. Select a baseline point and confirm the viscosity using Kanazawa line fitting.\n\n"
            "8. Click Optimize Cycle \"x\" to check the fitting results. Double-click any point to view fitting details. Repeat for all cycles, then click Run Dynamic Optimization (All Cycles) for global fitting.\n\n"
            "9. If viscosity changes within a cycle, use Manual Segmentation. Each segment will have independent Theta and viscosity settings.\n\n"
            "10. Export results using Combined Analysis (Image) or Combined Analysis (Excel).\n\n"
            "11. Click Reset Manual Splits to return to the default cycle view"
        )
        lbl = ctk.CTkLabel(self.guide_window, text=instructions_text, font=("Arial", 13), justify="left", wraplength=650)
        lbl.pack(pady=(0, 10), padx=25, anchor="w")

    def reset_manual_segmentation_state(self):
        self.saved_manual_splits = []
        self.manual_split_memory = {}
        self.manual_cycle_segment_groups = {}
        self.manual_cycle_viscosity_memory = {}

        if self.dynamic_window is not None and self.dynamic_window.winfo_exists():
            self.dynamic_window.destroy()
        self.dynamic_window = None

        if self.combined_window is not None and self.combined_window.winfo_exists():
            self.combined_window.destroy()
        self.combined_window = None

        if self.row_window is not None and self.row_window.winfo_exists():
            self.row_window.destroy()
        self.row_window = None

    def invalidate_full_run(self, event=None):
        self.full_run_completed = False

    def create_inputs_frame(self):
        self.input_container = ctk.CTkFrame(self.scroll_frame, fg_color="transparent")
        self.input_container.pack(pady=10, padx=10, fill="x", expand=True)

        self.left_col = ctk.CTkFrame(self.input_container, fg_color="transparent")
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 10))

        self.right_col = ctk.CTkFrame(self.input_container, fg_color="#262626", corner_radius=10)
        self.right_col.pack(side="right", fill="y", padx=(10, 0), ipadx=10, ipady=10)

        ctk.CTkLabel(self.left_col, text="Experiment Parameters", font=("Arial", 18, "bold")).pack(anchor="center", pady=(0, 15))

        self.params_grid = ctk.CTkFrame(self.left_col, fg_color="transparent")
        self.params_grid.pack(fill="x", expand=True)

        self.card_qcm = ctk.CTkFrame(self.params_grid, fg_color="#262626", corner_radius=10)
        self.card_qcm.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        self.card_echem = ctk.CTkFrame(self.params_grid, fg_color="#262626", corner_radius=10)
        self.card_echem.pack(side="left", fill="both", expand=True, padx=5)
        
        self.card_elec = ctk.CTkFrame(self.params_grid, fg_color="#262626", corner_radius=10)
        self.card_elec.pack(side="left", fill="both", expand=True, padx=(5, 0))

        ctk.CTkLabel(self.card_qcm, text="QCM Physical Parameters", font=("Arial", 15, "bold"), text_color="#3498db").pack(pady=(15, 0))
        ctk.CTkLabel(self.card_qcm, text="Sensor physical constants", font=("Arial", 11), text_color="gray").pack(pady=(0, 10))
        self._add_param_row(self.card_qcm, "Quartz Density [g/cm^3]", "2.648")
        self._add_param_row(self.card_qcm, "Quartz Viscosity [Pa]", "2.947e10")
        self._add_param_row(self.card_qcm, "Active Area [cm^2]", "0.785")
        self._add_param_row(self.card_qcm, "Sensitivity [Hz·cm^2/ug]", "56.5")

        ctk.CTkLabel(self.card_echem, text="Electrochemical", font=("Arial", 15, "bold"), text_color="#2ecc71").pack(pady=(15, 0))
        ctk.CTkLabel(self.card_echem, text="Reaction & electrode details", font=("Arial", 11), text_color="gray").pack(pady=(0, 10))
        self._add_param_row(self.card_echem, "Molecular Weight [g/mol]", "65")
        self._add_param_row(self.card_echem, "Number of Electrons", "2")
        self._add_param_row(self.card_echem, "Coverage (θ) [range: 0–1]", "1.0")

        ctk.CTkLabel(self.card_elec, text="Electrolyte", font=("Arial", 15, "bold"), text_color="#9b59b6").pack(pady=(15, 0))
        ctk.CTkLabel(self.card_elec, text="Fluid properties & correction", font=("Arial", 11), text_color="gray").pack(pady=(0, 10))
        self._add_param_row(self.card_elec, "Liquid Density [g/cm^3]", "1.3228")
        self._add_param_row(self.card_elec, "Ref. Liquid Viscosity [Pa·s]", "0.0032")

        ctk.CTkLabel(self.right_col, text="Harmonics Configuration", font=("Arial", 16, "bold")).pack(anchor="center", padx=15, pady=(15, 10))
        
        ctk.CTkLabel(self.right_col, text="Quartz crystal signal measured in air (Hz/ppm):", font=("Arial", 12, "bold"), text_color="#f1c40f").pack(anchor="center", padx=15, pady=(5, 10))

        headers_frame = ctk.CTkFrame(self.right_col, fg_color="transparent")
        headers_frame.pack(fill="x", padx=15)
        ctk.CTkLabel(headers_frame, text="", width=100).grid(row=0, column=0)
        ctk.CTkLabel(headers_frame, text="Ref F (Hz)", font=("Arial", 12, "bold"), width=130).grid(row=0, column=1, padx=5)
        ctk.CTkLabel(headers_frame, text="Ref D (ppm)", font=("Arial", 12, "bold"), width=130).grid(row=0, column=2, padx=5)

        self.harmonics_frame_inner = ctk.CTkScrollableFrame(self.right_col, fg_color="transparent", height=280)
        self.harmonics_frame_inner.pack(fill="both", expand=True, padx=10, pady=5)

    def _add_param_row(self, parent, text, val):
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.pack(fill="x", pady=12, padx=15)
        lbl = ctk.CTkLabel(row_frame, text=text + ":", font=("Arial", 12))
        lbl.pack(side="left")
        entry = ctk.CTkEntry(row_frame, width=95, justify="center")
        entry.insert(0, val)
        entry.pack(side="right")
        self.entries[text] = entry

    def update_harmonic_fields(self):
        self.invalidate_full_run()
        for widget in self.harmonics_widgets: widget.destroy()
        self.harmonics_widgets = []
        self.air_f_entries = {}; self.air_d_entries = {}
        
        current_row = 0
        
        for n in self.current_harmonics:
            lbl = ctk.CTkLabel(self.harmonics_frame_inner, text=f"Harmonic n={n}:", font=("Arial", 12), width=100, anchor="e")
            lbl.grid(row=current_row, column=0, padx=5, pady=5)
            self.harmonics_widgets.append(lbl)
            
            ef = ctk.CTkEntry(self.harmonics_frame_inner, width=130, fg_color="#333333", border_width=1)
            ef.insert(0, "0.0")
            ef.grid(row=current_row, column=1, padx=5, pady=5)
            ef.bind("<KeyRelease>", self.invalidate_full_run)
            self.harmonics_widgets.append(ef)
            self.air_f_entries[n] = ef
            
            ed = ctk.CTkEntry(self.harmonics_frame_inner, width=130, fg_color="#333333", border_width=1)
            ed.insert(0, "0.0")
            ed.grid(row=current_row, column=2, padx=5, pady=5)
            ed.bind("<KeyRelease>", self.invalidate_full_run)
            self.harmonics_widgets.append(ed)
            self.air_d_entries[n] = ed
            
            current_row += 1
            
        self.apply_reference_data()

    def parse_reference_file(self, file_path):
        try:
            df_ref = pd.read_excel(file_path)
            f_vals = {}
            d_vals = {}
            
            for col in df_ref.columns:
                col_str = str(col).strip()
                
                f_match = re.match(r'^[fF](\d+)', col_str)
                if f_match:
                    n = int(f_match.group(1))
                    if n % 2 != 0: 
                        valid_series = pd.to_numeric(df_ref[col], errors='coerce').dropna()
                        if not valid_series.empty:
                            f_vals[n] = float(valid_series.iloc[-1])
                
                d_match = re.match(r'^[dD](\d+)', col_str)
                if d_match:
                    n = int(d_match.group(1))
                    if n % 2 != 0:
                        valid_series = pd.to_numeric(df_ref[col], errors='coerce').dropna()
                        if not valid_series.empty:
                            d_vals[n] = float(valid_series.iloc[-1])
                        
            return {'f': f_vals, 'd': d_vals}
        except Exception as e:
            print(f"Error parsing reference file: {e}")
            return {'f': {}, 'd': {}}

    def apply_reference_data(self):
        source_data = self.raw_air_data
        
        for n in self.current_harmonics:
            if n in self.air_f_entries:
                val_f = source_data['f'].get(n, 0.0)
                if pd.isna(val_f): val_f = 0.0
                self.air_f_entries[n].delete(0, 'end')
                self.air_f_entries[n].insert(0, str(val_f))
                
            if n in self.air_d_entries:
                val_d = source_data['d'].get(n, 0.0)
                if pd.isna(val_d): val_d = 0.0
                self.air_d_entries[n].delete(0, 'end')
                self.air_d_entries[n].insert(0, str(val_d))

    def load_air_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.raw_air_data = self.parse_reference_file(file_path)
                self.apply_reference_data()
                
                base_name = os.path.basename(file_path)
                self.lbl_file_air.configure(text=self.truncate_string(base_name), text_color="white")
                self.log(f"Quartz in Air Data File loaded: {base_name}. Extracted last valid stabilization points.")
                self.invalidate_full_run()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_material_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.raw_material_data = self.parse_reference_file(file_path)
                
                base_name = os.path.basename(file_path)
                if self.lbl_file_material is not None:
                    self.lbl_file_material.configure(text=self.truncate_string(base_name), text_color="white")
                self.log(f"Quartz with Material Data File loaded: {base_name}. Extracted last valid stabilization points.")
                self.invalidate_full_run()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_coated_material_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.raw_coated_material_data = self.parse_reference_file(file_path)
                
                base_name = os.path.basename(file_path)
                if self.lbl_file_coated_material is not None:
                    self.lbl_file_coated_material.configure(text=self.truncate_string(base_name), text_color="white")
                self.log(f"Coated Quartz with Material Data File loaded: {base_name}. Extracted last valid stabilization points.")
                self.invalidate_full_run()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def load_qcmd_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.df = pd.read_excel(file_path)
                self.df_original = self.df.copy() 
                self.current_time_offset = 0.0    
                self.lbl_sync_ref.configure(text="")
                
                self.baseline_indices = None
                
                self.reset_manual_segmentation_state()
                
                self.invalidate_full_run()
                
                found_harmonics = []
                for col in self.df.columns:
                    match = re.match(r'^[fF](\d+)', str(col).strip())
                    if match:
                        n = int(match.group(1))
                        if n % 2 != 0:
                            found_harmonics.append(n)
                
                if found_harmonics:
                    min_n = min(found_harmonics)
                    max_n = max(found_harmonics)
                    self.current_harmonics = list(range(min_n, max_n + 2, 2))
                    self.log(f"Auto-detected harmonics from QCMD: {self.current_harmonics}")
                else:
                    self.current_harmonics = [3, 5, 7, 9, 11]
                    self.log("No valid F columns detected, defaulting to 3-11.")
                
                self.update_harmonic_fields()
                
                base_name = os.path.basename(file_path)
                self.filename_qcmd = os.path.splitext(base_name)[0]
                self.lbl_file_qcmd.configure(text=self.truncate_string(base_name), text_color="white")
                self.log(f"QCM-D File loaded: {base_name} with {len(self.df)} rows.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def load_echem_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if file_path:
            try:
                self.df_echem = pd.read_excel(file_path)
                self.df_echem_original = self.df_echem.copy() 
                self.current_time_offset = 0.0                
                self.lbl_sync_ref.configure(text="")
                
                self.baseline_indices = None
                self.reset_manual_segmentation_state()
                self.invalidate_full_run()
                
                base_name = os.path.basename(file_path)
                self.filename_echem = os.path.splitext(base_name)[0]
                self.lbl_file_echem.configure(text=self.truncate_string(base_name), text_color="white")
                self.log(f"Electrochemistry File loaded: {base_name} with {len(self.df_echem)} rows.")
            except Exception as e: messagebox.showerror("Error", str(e))

    def open_dynamic_window(self):
        if self.df is None or self.df_echem is None:
            messagebox.showwarning("Warning", "Please load BOTH QCM-D and EChem data first to detect cycles.")
            return

        success = self.prepare_theoretical_and_cycles()
        if not success: return

        segment_groups = self.build_cycle_segment_groups()

        if self.dynamic_window is None or not self.dynamic_window.winfo_exists():
            self.dynamic_window = DynamicParamsWindow(
                self,
                run_callback=lambda config: self.run_full_process(dynamic_config=config, first_cycle_only=False),
                segment_groups=segment_groups
            )
            self.dynamic_window.lift()
        else:
            self.dynamic_window.destroy()
            self.dynamic_window = DynamicParamsWindow(
                self,
                run_callback=lambda config: self.run_full_process(dynamic_config=config, first_cycle_only=False),
                segment_groups=segment_groups
            )
            self.dynamic_window.lift()

    def find_cycle_bounds_for_index(self, target_idx):
        if self.cycle_indices is None or len(self.cycle_indices) < 3:
            return None
        for i in range(0, len(self.cycle_indices) - 2, 2):
            idx_start = int(self.cycle_indices[i])
            idx_end = int(self.cycle_indices[i + 2])
            
            # Identify if this is the very last cycle in the list
            is_last_cycle = (i == len(self.cycle_indices) - 3)
            
            # Use strict less-than (<) for the end boundary to prevent overlap, 
            # unless it's the very last cycle where we must include the final point.
            if idx_start <= target_idx < idx_end or (is_last_cycle and target_idx == idx_end):
                return idx_start, idx_end, (i // 2) + 1
        return None

    def build_cycle_segment_groups(self, selected_cycle_num=None, selected_segments=None):
        local_groups = []

        if self.cycle_indices is not None and len(self.cycle_indices) >= 3:
            for i in range(0, len(self.cycle_indices) - 2, 2):
                cycle_num = (i // 2) + 1
                idx_start = int(self.cycle_indices[i])
                idx_end = int(self.cycle_indices[i + 2])

                if selected_cycle_num is not None and cycle_num == selected_cycle_num and selected_segments is not None:
                    cycle_segments = selected_segments
                else:
                    cycle_segments = self.manual_cycle_segment_groups.get(cycle_num, [(idx_start, idx_end)])

                normalized_segments = sorted(
                    [(int(seg_start), int(seg_end)) for seg_start, seg_end in cycle_segments],
                    key=lambda seg: seg[0]
                )
                local_groups.append({'label': f"#{cycle_num}", 'segments': normalized_segments})
        elif selected_segments is not None:
            cycle_label = f"#{selected_cycle_num}" if selected_cycle_num is not None else "#1"
            local_groups = [{'label': cycle_label, 'segments': selected_segments}]

        return local_groups

    def open_manual_segments_config(self, custom_segments, title_suffix="", save_key="global", segment_groups=None):
        if self.dynamic_window is not None and self.dynamic_window.winfo_exists():
            self.dynamic_window.destroy()
        self.dynamic_window = DynamicParamsWindow(
            self,
            run_callback=lambda config: self.run_full_process(dynamic_config=config, first_cycle_only=False),
            custom_segments=custom_segments,
            segment_groups=segment_groups,
            window_title=(f"Dynamic Parameters | {title_suffix}" if title_suffix else "Dynamic Parameters per Cycle")
        )
        self.dynamic_window.lift()

    def open_manual_segmentation_window(self, df_override=None, filename_override=None, scope="global", anchor_idx=None):
        if self.df is None or self.df_echem is None:
            messagebox.showwarning("Warning", "Please load BOTH QCM-D and EChem data first.")
            return

        required_cols = ["Graph_h_nm", "Graph_xi_nm"]
        current_working_df = getattr(self, "working_df", None)
        has_existing_optimized_view = current_working_df is not None and all(col in current_working_df.columns for col in required_cols)
        if not has_existing_optimized_view:
            success = self.prepare_theoretical_and_cycles()
            if not success:
                return

        target_df = self.working_df.copy() if df_override is None else df_override.copy()
        segment_groups = None
        selected_cycle_num = None
        title_suffix = filename_override if filename_override else "Global"
        save_key = f"{scope}:{title_suffix}"

        if scope == "row":
            cycle_info = self.find_cycle_bounds_for_index(anchor_idx)
            if cycle_info is None:
                messagebox.showwarning("Warning", "Could not resolve the current row to a detected cycle.", parent=self)
                return
            idx_start, idx_end, cycle_num = cycle_info
            target_df = self.working_df.loc[idx_start:idx_end].copy()
            title_suffix = f"{self.filename_qcmd}_Cycle_{cycle_num}"
            save_key = f"row_cycle:{cycle_num}"
            selected_cycle_num = cycle_num
        elif df_override is not None and self.working_df is not None and len(target_df) < len(self.working_df):
            scope = "local_cycle"
            save_key = f"local:{title_suffix}"
            if self.cycle_indices is not None and len(target_df) > 0:
                cycle_info = self.find_cycle_bounds_for_index(int(target_df.index[0]))
                if cycle_info is not None:
                    selected_cycle_num = cycle_info[2]

        global_f0 = {}
        global_d0 = {}
        if self.working_df is not None and not self.working_df.empty:
            for n in self.current_harmonics:
                col_f = f"f{n}"
                if col_f in self.working_df.columns:
                    global_f0[n] = self.working_df[col_f].iloc[0]
                col_d = f"D{n}" if f"D{n}" in self.working_df.columns else f"d{n}"
                if col_d in self.working_df.columns:
                    global_d0[n] = self.working_df[col_d].iloc[0]

        def handle_confirm(custom_segments, _x_vals):
            if selected_cycle_num is not None:
                self.manual_cycle_segment_groups[selected_cycle_num] = sorted(
                    [(int(seg_start), int(seg_end)) for seg_start, seg_end in custom_segments],
                    key=lambda seg: seg[0]
                )
            local_groups = self.build_cycle_segment_groups(selected_cycle_num, custom_segments)
            self.open_manual_segments_config(custom_segments, title_suffix=title_suffix, save_key=save_key, segment_groups=local_groups)

        OptimizationManualSegmentationWindow(
            self,
            df_override=target_df,
            title_suffix=title_suffix,
            save_key=save_key,
            on_confirm_callback=handle_confirm,
            global_f0=global_f0,
            global_d0=global_d0
        )

    def open_sync_window(self):
        if self.df is None or self.df_echem is None:
            messagebox.showwarning("Warning", "Please load BOTH QCM-D and Electro-chemical data first before syncing.")
            return

        if self.sync_window is None or not self.sync_window.winfo_exists():
            self.sync_window = TimeSyncWindow(self)
            self.sync_window.lift()
        else:
            self.sync_window.focus()

    def prepare_theoretical_and_cycles(self):
        if self.df is None or self.df_echem is None:
            return False
            
        self.working_df = self.df.copy()
        self.log("Calculating Theoretical Curve and Detecting Cycles...")
        try:
            Mw = float(self.entries["Molecular Weight [g/mol]"].get())
            n_elec = float(self.entries["Number of Electrons"].get())
            Area = float(self.entries["Active Area [cm^2]"].get())
            Cm = float(self.entries["Sensitivity [Hz·cm^2/ug]"].get())
            F_const = 96485.332
            
            time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'time/s']
            q_cols = ['Q', 'Charge', 'C', '(Q-Qo)', '(Q-Qo)/mC', 'Q/mC']
            e_cols = ['Ewe', 'Potential', 'Voltage', '<Ewe/V>', 'Ewe/V']
            
            t_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in time_cols)), None)
            q_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in q_cols)), None)
            e_col_ec = next((c for c in self.df_echem.columns if any(x in c for x in e_cols)), None)
            
            qcm_time_col = next((c for c in self.working_df.columns if any(x in c for x in time_cols)), None)

            if t_col_ec and q_col_ec and qcm_time_col:
                t_ec = pd.to_numeric(self.df_echem[t_col_ec], errors='coerce').fillna(0).to_numpy()
                q_raw = pd.to_numeric(self.df_echem[q_col_ec], errors='coerce').fillna(0).to_numpy()
                Q_coulombs = q_raw * 1e-3 if ("mC" in q_col_ec or "mc" in q_col_ec) else q_raw
                delta_f_theo_raw_all = ((Cm * 1e6) * Mw * Q_coulombs) / (n_elec * F_const * Area)
                
                e_raw_all = None
                if e_col_ec:
                    e_raw_all = pd.to_numeric(self.df_echem[e_col_ec], errors='coerce').fillna(0).to_numpy()
                
                t_qcm = pd.to_numeric(self.working_df[qcm_time_col], errors='coerce').fillna(0).to_numpy()
                indices_to_keep = []
                for t_val in t_qcm:
                    idx = (np.abs(t_ec - t_val)).argmin()
                    indices_to_keep.append(idx)
                
                f_theo_synced = delta_f_theo_raw_all[indices_to_keep]
                
                if e_raw_all is not None:
                      e_synced = e_raw_all[indices_to_keep]
                      self.working_df["E_we_V"] = e_synced
                      
                      prominence_val = (np.max(e_synced) - np.min(e_synced)) * 0.15
                      distance_val = max(5, len(e_synced) // 200)
                      peaks, _ = find_peaks(e_synced, prominence=prominence_val, distance=distance_val)
                      
                      pois = [0]
                      all_bounds = [0] + list(peaks) + [len(e_synced)-1]
                      for i in range(len(all_bounds)-1):
                          idx_start = all_bounds[i]
                          idx_end = all_bounds[i+1]
                          if idx_end > idx_start:
                              segment = e_synced[idx_start:idx_end]
                              idx_valley = idx_start + np.argmin(segment)
                              if idx_valley != idx_start and idx_valley not in pois:
                                  pois.append(idx_valley)
                          if all_bounds[i+1] not in pois:
                              pois.append(all_bounds[i+1])
                              
                      self.cycle_indices = np.array(pois)
                      self.log(f">>> CP Cycle Detection: Found {len(peaks)} main peaks. Total table rows: {len(self.cycle_indices)//2}.")
                else:
                      self.log(">>> Warning: E_we_V not found, cycle detection might fail.")
                      self.cycle_indices = np.array([0, len(self.working_df)-1])

                all_norm_f = []
                for n in self.current_harmonics:
                    col_f = f"f{n}"
                    if col_f in self.working_df.columns:
                        raw_f = self.working_df[col_f].to_numpy()
                        norm_f = (raw_f - raw_f[0]) / n
                        all_norm_f.append(norm_f)
                
                if all_norm_f:
                    # Select the mathematically maximum value across all harmonics for each time point
                    qcm_target_curve = np.max(all_norm_f, axis=0)
                else:
                    qcm_target_curve = np.zeros_like(f_theo_synced)

                f_theo_calibrated = self.calibrate_theoretical_data(t_qcm, f_theo_synced, qcm_target_curve)

                self.working_df["Theo_Calibrated_Active"] = f_theo_calibrated
                self.working_df["F_Calibrated_View"] = f_theo_calibrated 
                return True
            else:
                messagebox.showerror("Error", "Could not find Time/Charge columns in files.")
                return False

        except Exception as e:
            self.log(f"Error in Cycle Calculation: {e}")
            messagebox.showerror("Error", f"Cycle detection failed. Check parameters: {e}")
            return False

    def calculate_optimization_loop(self, p, calc_tol, calc_pop, dynamic_config=None, df_subset=None, progress_callback=None):
        harmonics = self.current_harmonics
        working_data = self.working_df.copy() if df_subset is None else df_subset.copy()
        total_rows = len(working_data)
        
        time_col = None
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in working_data.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                time_col = col
                break
        
        if dynamic_config and time_col is None:
             self.log("Warning: Could not find Time column. Using default parameters for all rows.")
             dynamic_config = None

        if "Theo_Calibrated_Active" in working_data.columns:
            theo_col = "Theo_Calibrated_Active"
        else:
            messagebox.showerror("Error", "Calibrated Theoretical Data is missing. Check EChem file.")
            return None, None, None, None, None, None, None, None, 0
        
        res_h, res_xi, res_cutoff, res_status = [], [], [], []
        res_curves = [] 
        res_exp_3rd = [] 
        res_visc = [] 
        res_theta = []
        
        successful_points = 0
        last_valid_xi = None
        candidate_xi = None
        candidate_idx = None

        visc_ranges = dynamic_config.get('visc', []) if dynamic_config else []
        theta_ranges = dynamic_config.get('theta', []) if dynamic_config else []

        for i, (index, row) in enumerate(working_data.iterrows()):
            if self.stop_flag:
                self.log(">>> Process Stopped by user.")
                return None, None, None, None, None, None, None, None, 0
            
            if i % 5 == 0:
                if progress_callback:
                    progress_callback(i + 1, total_rows)
                else:
                    self.lbl_progress_text.configure(text=f"Processing: {i+1} / {total_rows}")
                    self.update()

            current_visc_liq = p["visc_liq"] 
            current_theta = p["theta"]

            if time_col and dynamic_config:
                t_val = row[time_col]
                for r in visc_ranges:
                    if r['start'] <= t_val <= r['end']:
                        current_visc_liq = r['val']
                        break
                for r in theta_ranges:
                      if r['start'] <= t_val <= r['end']:
                        current_theta = r['val']
                        break
            
            res_visc.append(current_visc_liq)
            res_theta.append(current_theta)

            row_dn_vals = []
            for n in harmonics:
                f_air = p["air_f"][n]
                val_dn = 1e6 * np.sqrt(current_visc_liq / (np.pi * p["rho_liq"] * f_air))
                row_dn_vals.append(val_dn)
            
            row_dn_um = np.array(row_dn_vals)
            row_dn_meters = row_dn_um * 1e-6

            try:
                current_df_vals = []
                current_dw_vals = []
                valid_row = True
                
                for n in harmonics:
                    col_f = f"f{n}"
                    col_d = f"D{n}" if f"D{n}" in working_data.columns else f"d{n}"
                    if col_f not in working_data.columns or col_d not in working_data.columns:
                        valid_row = False; break

                    val_f = row[col_f]; val_d = row[col_d]
                    f_air = p["air_f"][n]; d_air = p["air_d"][n]

                    delta_f = val_f - f_air
                    denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
                    num_f = (delta_f / n) - row[theo_col] 
                    calc_df = num_f / denominator 
                    
                    term_w_liq = (val_d * 1e-6) * (val_f / n)
                    term_w_air = (d_air * 1e-6) * (f_air / n)
                    num_w = term_w_liq - term_w_air
                    calc_dw = num_w / denominator 

                    current_df_vals.append(calc_df)
                    current_dw_vals.append(calc_dw)

                if not valid_row:
                    res_h.append(None); res_xi.append(None); res_cutoff.append(None); res_status.append("Missing Cols")
                    res_curves.append(None); res_exp_3rd.append(None)
                    continue

                target_df = np.array(current_df_vals) * 1e9
                target_dw = np.array(current_dw_vals) * 1e9

                if 3 in harmonics:
                    idx_3 = harmonics.index(3)
                    val_dn_3 = row_dn_um[idx_3] 
                    val_w_3 = target_dw[idx_3]
                    res_exp_3rd.append((val_dn_3, val_w_3))
                else:
                    res_exp_3rd.append(None)

                def objective_function(params_opt):
                    qsi_nm, h_nm = params_opt
                    qsi = qsi_nm * 1e-9; h = h_nm * 1e-9  
                    w_model, f_model = dalta_model(row_dn_meters, current_theta, qsi, h, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])
                    if w_model is None: return 1e9 
                    
                    weights_list = [10 if k < 3 else 1.0 for k in range(len(harmonics))]
                    f_weights_arr = np.array(weights_list)
                    
                    error_w = np.sum((w_model - target_dw)**2)
                    error_f = np.sum(((f_model - target_df)**2) * f_weights_arr)
                    return (100 * error_w) + error_f

                bounds = [(0.1, 200.0), (0.1, 1500.0)]
                cutoff_val = find_cutoff(row_dn_um, target_dw, current_theta, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])

                best_physics_qsi_nm = None
                best_physics_h_nm = None
                status = "Failed"
                
                for attempt in range(3):
                    result = differential_evolution(objective_function, bounds, strategy='best1bin', maxiter=10000, popsize=calc_pop, tol=calc_tol)
                    
                    if not result.success:
                        continue
                        
                    cand_qsi, cand_h = result.x
                    
                    if cutoff_val is not None and cand_h < cutoff_val:
                        best_physics_qsi_nm = 0.1
                        best_physics_h_nm = cand_h
                        status = "Cutoff Correction"
                        break
                        
                    if cand_qsi <= 50.0 and cand_qsi <= cand_h:
                        best_physics_qsi_nm = cand_qsi
                        best_physics_h_nm = cand_h
                        
                        target = candidate_xi if candidate_xi is not None else last_valid_xi
                        
                        if target is None:
                            status = "Failed_Stability"
                        else:
                            if abs(cand_qsi - target) <= 0.20 * target:
                                status = "Optimized"
                                break
                            else:
                                status = "Failed_Stability"
                                
                final_status = "Failed"
                final_qsi = None
                final_h = None
                
                if status == "Cutoff Correction":
                    final_qsi = 0.1
                    final_h = best_physics_h_nm
                    final_status = "Cutoff Correction"
                    last_valid_xi = 0.1  
                    
                    if candidate_idx is not None:
                        res_h[candidate_idx] = np.nan
                        res_xi[candidate_idx] = np.nan
                        res_status[candidate_idx] = "Failed_Validation"
                        candidate_xi, candidate_idx = None, None
                        
                elif status == "Optimized":
                    final_qsi = best_physics_qsi_nm
                    final_h = best_physics_h_nm
                    final_status = "Optimized"
                    
                    if candidate_idx is not None:
                        res_status[candidate_idx] = "Optimized"
                        candidate_xi, candidate_idx = None, None
                        
                    last_valid_xi = final_qsi
                    
                elif status == "Failed_Stability":
                    final_qsi = best_physics_qsi_nm
                    final_h = best_physics_h_nm
                    
                    if candidate_idx is not None:
                        res_h[candidate_idx] = np.nan
                        res_xi[candidate_idx] = np.nan
                        res_status[candidate_idx] = "Failed_Validation"
                        candidate_xi, candidate_idx = None, None
                        
                        if last_valid_xi is not None and abs(final_qsi - last_valid_xi) <= 0.20 * last_valid_xi:
                            final_status = "Optimized"
                            last_valid_xi = final_qsi
                        else:
                            final_status = "Candidate"
                    else:
                        final_status = "Candidate"
                        
                    if final_status == "Candidate":
                        candidate_xi = final_qsi
                        candidate_idx = len(res_h) 
                
                if final_status in ["Cutoff Correction", "Optimized", "Candidate"]:
                    res_h.append(final_h)
                    res_xi.append(final_qsi)
                    res_cutoff.append(cutoff_val)
                    res_status.append(final_status)
                    successful_points += 1
                    
                    dn_smooth_m = np.linspace(0.003, 0.3, 500) * 1e-6
                    model_w_curve, _ = dalta_model(dn_smooth_m, current_theta, final_qsi * 1e-9, final_h * 1e-9, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])
                    res_curves.append(np.column_stack((dn_smooth_m, model_w_curve)) if model_w_curve is not None else None)
                else:
                    res_h.append(np.nan)
                    res_xi.append(np.nan)
                    res_cutoff.append(cutoff_val)
                    res_status.append("Failed")
                    res_curves.append(None)

            except Exception as e:
                res_h.append(np.nan); res_xi.append(np.nan); res_cutoff.append(None); res_status.append("Error")
                res_curves.append(None); res_exp_3rd.append(None)
        
        if candidate_idx is not None:
            res_h[candidate_idx] = np.nan
            res_xi[candidate_idx] = np.nan
            res_status[candidate_idx] = "Failed_Validation"

        return res_h, res_xi, res_cutoff, res_status, res_curves, res_exp_3rd, res_visc, res_theta, successful_points

    def calibrate_theoretical_data(self, t_arr, f_theo_arr, f_qcm_avg_arr):
        try:
            if len(f_theo_arr) < 3 or self.cycle_indices is None or len(self.cycle_indices) < 3: 
                return f_theo_arr 
            
            pois = self.cycle_indices
            f_calibrated = np.copy(f_theo_arr)

            for i in range(0, len(pois) - 2, 2):
                idx_start = pois[i] 
                idx_mid = pois[i+1]
                idx_end = pois[i+2]

                qcm_val_end = f_qcm_avg_arr[idx_end]
                theo_val_end = f_theo_arr[idx_end]
                offset = qcm_val_end - theo_val_end
                
                f_calibrated[idx_mid : idx_end + 1] = f_theo_arr[idx_mid : idx_end + 1] + offset
                
                new_mid_val = f_calibrated[idx_mid]

                qcm_val_start = f_qcm_avg_arr[idx_start] 
                
                n_points = idx_mid - idx_start
                if n_points > 0:
                    linear_segment = np.linspace(qcm_val_start, new_mid_val, n_points + 1)
                    f_calibrated[idx_start : idx_mid + 1] = linear_segment

            self.log(f">>> Calibration Logic Applied: Used pre-calculated {len(pois)} POIs from E.")
            return f_calibrated

        except Exception as e:
            self.log(f"Error in Calibration: {e}")
            return f_theo_arr

    def run_full_process(self, dynamic_config=None, first_cycle_only=False):
        if self.is_running:
            self.log(">>> Stopping current process to restart...")
            self.stop_flag = True
            self.after(200, lambda: self.run_full_process(dynamic_config, first_cycle_only))
            return

        if self.df is None:
            messagebox.showwarning("Warning", "Please load a QCM-D file first.")
            return

        if self.df_echem is None:
            messagebox.showerror("Error", "EChem Data is required for calculation. Please load it.")
            return

        p = self.get_params()
        if not p: return
        
        if first_cycle_only:
            res_choice = "Low (Fast) [Forced]"
            calc_tol, calc_pop = 0.1, 10
        else:
            res_choice = self.res_var.get()
            if "High" in res_choice: calc_tol, calc_pop = 0.001, 40
            elif "Low" in res_choice: calc_tol, calc_pop = 0.1, 10
            else: calc_tol, calc_pop = 0.01, 20

        mode_str = "First Cycle Only" if first_cycle_only else ("Dynamic Params" if dynamic_config else "Standard")
        self.log(f"Starting Process... ({res_choice} | {mode_str})")
        if dynamic_config and not first_cycle_only:
            v_len = len(dynamic_config.get('visc', []))
            t_len = len(dynamic_config.get('theta', []))
            self.log(f" > Config: {v_len} Visc Ranges, {t_len} Theta Ranges")
        
        self.is_running = True
        self.stop_flag = False
        start_time = time.time()

        if not self.prepare_theoretical_and_cycles():
            self.is_running = False
            return

        if first_cycle_only:
            if self.cycle_indices is not None and len(self.cycle_indices) >= 3:
                end_idx = self.cycle_indices[2]
                self.working_df = self.working_df.iloc[:end_idx + 1].copy()
                self.cycle_indices = self.cycle_indices[:3] 
                self.log(f">>> First Cycle Only mode: Slicing data up to index {end_idx}.")
            else:
                self.log(">>> First Cycle Only mode: Could not detect a full cycle. Running on first 100 points as fallback.")
                self.working_df = self.working_df.iloc[:100].copy()

        res_h, res_xi, res_cutoff, res_status, res_curves, res_exp_3rd, res_visc, res_theta, successful_points = self.calculate_optimization_loop(p, calc_tol, calc_pop, dynamic_config)
        
        self.is_running = False 

        if res_h is None: 
            self.log("Process Stopped or Failed.")
            return 

        self.working_df["Optimized_Height_nm"] = res_h
        self.working_df["Optimized_Qsi_nm"] = res_xi
        
        self.working_df["Graph_h_nm"] = self.working_df["Optimized_Height_nm"].interpolate(method='linear')
        self.working_df["Graph_xi_nm"] = self.working_df["Optimized_Qsi_nm"].interpolate(method='linear')

        self.working_df["Cutoff_nm"] = res_cutoff
        self.working_df["Fit_Status"] = res_status
        self.working_df["Used_Viscosity_Pa_s"] = res_visc
        self.working_df["Used_Theta"] = res_theta
        
        total_pts = len(self.working_df)
        success_rate = (successful_points / total_pts) * 100 if total_pts > 0 else 0
        self.log("-" * 40)
        self.log(f">>> Optimization Success Rate: {success_rate:.2f}% ({successful_points}/{total_pts})")
        self.log("-" * 40)

        duration = time.time() - start_time
        self.lbl_progress_text.configure(text="Finished!")
        self.log(f"Done! {len(self.working_df)} rows in {duration:.2f}s.")
        
        if not first_cycle_only:
            self.full_run_completed = True
            
        current_cycle_times = None
        if self.cycle_indices is not None and len(self.cycle_indices) > 0:
            try:
                possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'time/s']
                qcm_time_col = next((c for c in self.working_df.columns if any(x in c for x in possible_time_cols)), None)
                if qcm_time_col:
                    times = pd.to_numeric(self.working_df[qcm_time_col], errors='coerce').fillna(0).to_numpy()
                    filtered_indices = self.cycle_indices[::2]
                    current_cycle_times = times[filtered_indices]
            except Exception as e:
                print("Error extracting cycle times:", e)
        
        fname_override = self.filename_qcmd + "_Dynamic_Params" if dynamic_config else self.filename_qcmd
        self.open_combined_graph_window(cycle_times=current_cycle_times, cycle_indices=self.cycle_indices, filename_override=fname_override)

    def open_combined_graph_window(self, cycle_times=None, cycle_indices=None, filename_override=None, df_override=None):
        target_df = df_override if df_override is not None else self.working_df
        if target_df is None: return
        
        global_f0 = {}
        global_d0 = {}
        if self.working_df is not None and not self.working_df.empty:
            for n in self.current_harmonics:
                col_f = f"f{n}"
                if col_f in self.working_df.columns:
                    global_f0[n] = self.working_df[col_f].iloc[0]
                col_d = f"D{n}" if f"D{n}" in self.working_df.columns else f"d{n}"
                if col_d in self.working_df.columns:
                    global_d0[n] = self.working_df[col_d].iloc[0]

        def manual_callback(df_view=target_df.copy(), fname=(filename_override if filename_override else self.filename_qcmd), is_local=(df_override is not None)):
            if self.combined_window is not None and self.combined_window.winfo_exists():
                self.combined_window.destroy()
                self.combined_window = None
            self.open_manual_segmentation_window(
                df_override=df_view,
                filename_override=fname,
                scope=("local_cycle" if is_local else "global")
            )
        
        if self.combined_window is None or not self.combined_window.winfo_exists():
            self.combined_window = CombinedGraphWindow(target_df, self.current_harmonics, self.filename_qcmd, self.plot_specific_row, manual_callback, cycle_times=cycle_times, cycle_indices=cycle_indices, filename_override=filename_override, global_f0=global_f0, global_d0=global_d0)
            self.combined_window.lift()
        else:
            self.combined_window.destroy()
            self.combined_window = CombinedGraphWindow(target_df, self.current_harmonics, self.filename_qcmd, self.plot_specific_row, manual_callback, cycle_times=cycle_times, cycle_indices=cycle_indices, filename_override=filename_override, global_f0=global_f0, global_d0=global_d0)

    def open_row_manual_segmentation(self, row_idx):
        if self.row_window is not None and self.row_window.winfo_exists():
            self.row_window.destroy()
            self.row_window = None
        self.open_manual_segmentation_window(
            scope="row",
            anchor_idx=row_idx,
            filename_override=f"{self.filename_qcmd}_Row_{row_idx}"
        )

    def log(self, message):
        print(message)
        self.update_idletasks()

    def get_params(self):
        try:
            air_f = {}; air_d = {}
            for n in self.current_harmonics:
                if n not in self.air_f_entries or n not in self.air_d_entries:
                    messagebox.showerror("Error", f"Harmonic {n} UI elements are missing.")
                    return None
                
                vf = self.air_f_entries[n].get(); vd = self.air_d_entries[n].get()
                if not vf or not vd: 
                    messagebox.showerror("Error", f"Air reference values for Harmonic n={n} cannot be empty.")
                    return None
                
                air_f[n] = float(vf); air_d[n] = float(vd)
            
            return {
                "rho_liq": float(self.entries["Liquid Density [g/cm^3]"].get()) * 1e3,
                "rho_quartz": float(self.entries["Quartz Density [g/cm^3]"].get()) * 1e3,
                "visc_liq": float(self.entries["Ref. Liquid Viscosity [Pa·s]"].get()),
                "visc_quartz": float(self.entries["Quartz Viscosity [Pa]"].get()),
                "theta": float(self.entries["Coverage (θ) [range: 0–1]"].get()),
                "air_f": air_f, "air_d": air_d
            }
        except KeyError as e: 
            messagebox.showerror("Error", f"Missing UI parameter mapping: {e}")
            return None
        except ValueError as e: 
            messagebox.showerror("Error", f"Invalid numerical parameters.\nPlease check your inputs.\nDetails: {e}")
            return None

    def plot_specific_row(self, target_idx=None):
        if self.working_df is None: return
        
        idx = -1
        if target_idx is not None: 
            idx = target_idx
        else:
            return 
            
        if idx < 0 or idx >= len(self.working_df): return
        p = self.get_params()
        if not p: return

        row_data = self.working_df.loc[idx]
        best_h_nm = row_data.get("Optimized_Height_nm", 0)
        best_qsi_nm = row_data.get("Optimized_Qsi_nm", 0)
        
        row_visc = row_data.get("Used_Viscosity_Pa_s")
        if pd.notna(row_visc):
            p["visc_liq"] = float(row_visc)
        
        row_theta = row_data.get("Used_Theta")
        if pd.notna(row_theta):
            p["theta"] = float(row_theta)
        
        if pd.isna(best_h_nm) or pd.isna(best_qsi_nm) or best_h_nm is None:
            messagebox.showerror("Error", f"Row {idx}: Invalid optimization results (failed fit).")
            return
        
        time_val = "N/A"
        possible_time_cols = ['time', 'Time', 't', 'sec', 'seconds', 'זמן', 'Time [sec]', 'AbsTime', 'RelTime']
        for col in self.working_df.columns:
            if any(t.lower() in col.lower() for t in possible_time_cols):
                val = row_data[col]
                time_val = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                break

        harmonics = self.current_harmonics
        
        if "Theo_Calibrated_Active" in self.working_df.columns:
            theo_col = "Theo_Calibrated_Active"
        else:
            theo_col = [c for c in self.working_df.columns if "theo" in c.lower()][0]

        exp_df_vals = []; exp_dw_vals = []
        dn_calculated_vals = []
        for n in harmonics:
            f_air = p["air_f"][n]
            val_dn = 1e6 * np.sqrt(p["visc_liq"] / (np.pi * p["rho_liq"] * f_air))
            dn_calculated_vals.append(val_dn)
        dn_calculated_um = np.array(dn_calculated_vals)

        for i, n in enumerate(harmonics):
            val_f = row_data[f"f{n}"]; val_d = row_data.get(f"D{n}", row_data.get(f"d{n}"))
            f_air = p["air_f"][n]; d_air = p["air_d"][n]
            delta_f = val_f - f_air
            denominator = (p["rho_liq"] / 1000.0) * ((f_air / n) ** 2)
            num_f = (delta_f / n) - row_data[theo_col] 
            calc_df = num_f / denominator 
            term_w_liq = (val_d * 1e-6) * (val_f / n)
            term_w_air = (d_air * 1e-6) * (f_air / n)
            num_w = term_w_liq - term_w_air
            calc_dw = num_w / denominator
            exp_df_vals.append(calc_df * 1e9); exp_dw_vals.append(calc_dw * 1e9)

        dn_smooth_m = np.linspace(0.003, 0.3, 500) * 1e-6
        model_w_curve, model_f_curve = dalta_model(dn_smooth_m, p["theta"], best_qsi_nm * 1e-9, best_h_nm * 1e-9, p["rho_liq"], p["rho_quartz"], p["visc_quartz"])
        
        if model_f_curve is None or model_w_curve is None:
            messagebox.showerror("Error", f"Row {idx}: Physics model failed to calculate curve (values diverged).")
            return

        kana_w, kana_f = model_Kanazawa_line(dn_smooth_m, p["rho_quartz"], p["visc_quartz"])

        col_n = harmonics
        col_dn = list(dn_calculated_um)
        col_df = exp_df_vals
        col_dw = exp_dw_vals
        col_model_dn = list(dn_smooth_m * 1e6) 
        col_model_f = list(model_f_curve)
        col_model_w = list(model_w_curve)

        param_names = ['ρ_L [g/cm^3]', 'η_L [Pa·s]', 'ρ_q [g/cm^3]', 'μ_q [Pa]', 'h [nm]', 'ξ [nm]', 'θ [0–1]']
        param_values = [p["rho_liq"] / 1000.0, p["visc_liq"], p["rho_quartz"] / 1000.0, p["visc_quartz"], best_h_nm, best_qsi_nm, p["theta"]]
        
        max_len = max(len(col_n), len(param_names), len(col_model_dn))
        def pad_list(l, length): return l + [None] * (length - len(l))
        
        export_data_dict = {
            'n (Exp)': pad_list(col_n, max_len), 'δ_µm (Exp)': pad_list(col_dn, max_len),
            'Y_Δf_1e9 (Exp)': pad_list(col_df, max_len), 'Y_ΔW_1e9 (Exp)': pad_list(col_dw, max_len),
            ' | ': [None] * max_len,
            'Model_δ_µm': pad_list(col_model_dn, max_len), 'Model_Δf': pad_list(col_model_f, max_len), 'Model_ΔW': pad_list(col_model_w, max_len),
            ' || ': [None] * max_len,
            'Parameter': pad_list(param_names, max_len), 'Value': pad_list(param_values, max_len)
        }

        if self.row_window is None or not self.row_window.winfo_exists():
            self.row_window = RowGraphWindow(idx, time_val, best_h_nm, best_qsi_nm,
                                             dn_smooth_m, model_f_curve, model_w_curve,
                                             dn_calculated_um, exp_df_vals, exp_dw_vals,
                                             kana_f, kana_w,
                                             self.filename_qcmd, export_data_dict, lambda row_idx=idx: self.open_row_manual_segmentation(row_idx))
            self.row_window.lift()
        else:
            self.row_window.destroy()
            self.row_window = RowGraphWindow(idx, time_val, best_h_nm, best_qsi_nm,
                                             dn_smooth_m, model_f_curve, model_w_curve,
                                             dn_calculated_um, exp_df_vals, exp_dw_vals,
                                             kana_f, kana_w,
                                             self.filename_qcmd, export_data_dict, lambda row_idx=idx: self.open_row_manual_segmentation(row_idx))

    def on_closing(self):
        self.stop_flag = True
        self.destroy()
        self.quit()

if __name__ == "__main__":
    app = PhysicsOptimizerApp()
    app.mainloop()
