# EIS Measurement System Using a Randles Equivalent Circuit

LTspice simulation and Python analysis of an electrochemical impedance
spectroscopy (EIS) measurement. A stepped-sine current (50 Hz to 10 kHz)
drives a Randles circuit (Rs = 0.01 ohm, Rp = 0.20 ohm, Cp = 1 mF).
A Python script computes Z = V/I from the FFT and plots Nyquist and Bode graphs.

## Structure
- simulation/ : LTspice schematic
- data/ : exported FFT data and result CSV
- scripts/ : nyquist_plot_n004(1).py
- results/ : plots

Author: Adithya Pushpavanam, M.Sc. Embedded Systems, TU Chemnitz
