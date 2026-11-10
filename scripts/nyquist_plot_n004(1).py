from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. FILE PATH
# =========================================================

# Paths are relative to the repository, so the script runs from any computer:
#   data/     -> LTspice FFT export and result CSV
#   results/  -> saved plots
BASE = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE / "data"
FOLDER = BASE / "results"
FOLDER.mkdir(exist_ok=True)

INPUT_FILE = DATA_FOLDER / "Draft5_n004.txt"
OUTPUT_CSV = DATA_FOLDER / "simulation_theory_results_n004.csv"

VOLTAGE_PHASE_OFFSET = 180


# =========================================================
# 2. FREQUENCIES
# =========================================================

frequency = np.array([50, 100, 200, 500, 1000, 2000, 5000, 10000], dtype=float)


# =========================================================
# 3. RANDLES CIRCUIT PARAMETERS
# =========================================================

Rs = 0.01
Rp = 0.20
Cp = 1000e-6


# =========================================================
# 4. FREQUENCY LABEL
# =========================================================


def make_label(f):

    if f >= 1000:
        return f"{f / 1000:g} kHz"

    return f"{f:g} Hz"


# Only the 8 simulated points are real results. The line between
# them is an interpolation (smoothing) for display only.
try:
    from scipy.interpolate import PchipInterpolator
except ImportError:
    PchipInterpolator = None


def smooth_curve(values, points=400):

    log_f = np.log10(frequency)
    log_f_smooth = np.linspace(log_f[0], log_f[-1], points)

    if PchipInterpolator is not None:
        smooth_values = PchipInterpolator(log_f, values)(log_f_smooth)
    else:
        smooth_values = np.interp(log_f_smooth, log_f, values)

    return 10**log_f_smooth, smooth_values


# =========================================================
# 5. READ LTSPICE FFT DATA
# =========================================================


def read_ltspice_fft(filename):

    if not filename.exists():
        raise FileNotFoundError(f"File not found:\n{filename}")

    text = filename.read_text(encoding="latin-1", errors="ignore")

    fft_frequencies = []
    voltages = []
    currents = []

    pattern = re.compile(
        r"\(\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*dB\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*[°º]?\s*\)"
    )

    for line in text.splitlines():

        columns = line.split()

        if not columns:
            continue

        try:
            current_frequency = float(columns[0])
        except ValueError:
            continue

        fft_values = pattern.findall(line)

        if len(fft_values) < 2:
            continue

        voltage_db = float(fft_values[0][0])
        voltage_phase = float(fft_values[0][1]) + VOLTAGE_PHASE_OFFSET

        current_db = float(fft_values[1][0])
        current_phase = float(fft_values[1][1])

        voltage_magnitude = 10 ** (voltage_db / 20)
        current_magnitude = 10 ** (current_db / 20)

        voltage_complex = voltage_magnitude * np.exp(1j * np.radians(voltage_phase))

        current_complex = current_magnitude * np.exp(1j * np.radians(current_phase))

        fft_frequencies.append(current_frequency)
        voltages.append(voltage_complex)
        currents.append(current_complex)

    if len(fft_frequencies) == 0:
        raise ValueError("No valid FFT data found.")

    return (np.array(fft_frequencies), np.array(voltages), np.array(currents))


# =========================================================
# 6. CALCULATE SIMULATED IMPEDANCE
# =========================================================


def calculate_simulated_impedance(fft_frequencies, voltages, currents):

    simulated_z = []

    for target_frequency in frequency:

        index = np.argmin(np.abs(fft_frequencies - target_frequency))

        impedance = voltages[index] / currents[index]

        simulated_z.append(impedance)

    return np.array(simulated_z)


# =========================================================
# 7. CALCULATE THEORETICAL RANDLES IMPEDANCE
# =========================================================


def calculate_randles_impedance(selected_frequencies):

    omega = 2 * np.pi * selected_frequencies
    Z = Rs + Rp / (1 + 1j * omega * Rp * Cp)

    return Z


# =========================================================
# 8. PRINT RESULTS
# =========================================================


def print_results(simulated_z, theoretical_z):

    print("\nSimulation and theoretical Randles results")

    print("=" * 128)

    print(
        f"{'Freq.':<9}"
        f"{'Sim Re(Z)':<13}"
        f"{'Sim Im(Z)':<13}"
        f"{'Sim |Z|':<13}"
        f"{'Sim Phase':<13}"
        f"{'Theory Re(Z)':<15}"
        f"{'Theory Im(Z)':<15}"
        f"{'Theory |Z|':<15}"
        f"{'Theory Phase':<15}"
    )

    print("-" * 128)

    for f, z_sim, z_theory in zip(frequency, simulated_z, theoretical_z):

        print(
            f"{f:<9.0f}"
            f"{z_sim.real:<13.6f}"
            f"{z_sim.imag:<13.6f}"
            f"{abs(z_sim):<13.6f}"
            f"{np.degrees(np.angle(z_sim)):<13.3f}"
            f"{z_theory.real:<15.6f}"
            f"{z_theory.imag:<15.6f}"
            f"{abs(z_theory):<15.6f}"
            f"{np.degrees(np.angle(z_theory)):<15.3f}"
        )


# =========================================================
# 9. SAVE RESULTS TO CSV
# =========================================================


def save_results(simulated_z, theoretical_z):

    data = np.column_stack(
        (
            frequency,
            simulated_z.real,
            simulated_z.imag,
            np.abs(simulated_z),
            np.degrees(np.angle(simulated_z)),
            theoretical_z.real,
            theoretical_z.imag,
            np.abs(theoretical_z),
            np.degrees(np.angle(theoretical_z)),
        )
    )

    header = (
        "Frequency_Hz,"
        "Sim_Re_Z_Ohm,"
        "Sim_Im_Z_Ohm,"
        "Sim_Magnitude_Ohm,"
        "Sim_Phase_Degree,"
        "Theory_Re_Z_Ohm,"
        "Theory_Im_Z_Ohm,"
        "Theory_Magnitude_Ohm,"
        "Theory_Phase_Degree"
    )

    np.savetxt(OUTPUT_CSV, data, delimiter=",", header=header, comments="", fmt="%.8f")

    print(f"\nResults saved as:\n{OUTPUT_CSV}")


# =========================================================
# 10. NYQUIST PLOT
# =========================================================


def plot_nyquist(simulated_z, theoretical_z):

    plt.figure(figsize=(10, 7))

    _, smooth_re = smooth_curve(simulated_z.real)
    _, smooth_im = smooth_curve(simulated_z.imag)

    plt.plot(smooth_re, -smooth_im, color="tab:blue", linewidth=1.5)

    plt.scatter(
        simulated_z.real,
        -simulated_z.imag,
        marker="o",
        s=45,
        color="tab:blue",
        label="LTspice simulation (8 points)",
        zorder=3,
    )

    smooth_frequency = np.logspace(np.log10(50), np.log10(10000), 500)

    smooth_theoretical_z = calculate_randles_impedance(smooth_frequency)

    plt.plot(
        smooth_theoretical_z.real,
        -smooth_theoretical_z.imag,
        linestyle="--",
        linewidth=2,
        color="tab:orange",
        label="Theoretical Randles circuit",
    )

    plt.scatter(theoretical_z.real, -theoretical_z.imag, marker="s", s=55, color="tab:orange")

    for f, z in zip(frequency, simulated_z):

        plt.annotate(
            make_label(f), (z.real, -z.imag), xytext=(7, 7), textcoords="offset points", fontsize=8
        )

    for f, z in zip(frequency, theoretical_z):

        plt.annotate(
            make_label(f),
            (z.real, -z.imag),
            xytext=(7, -15),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel("Re(Z) [Ω]")

    plt.ylabel("-Im(Z) [Ω]")

    plt.title("Nyquist Plot: LTspice Simulation vs Theoretical Randles Circuit")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = FOLDER / "nyquist_simulation_vs_randles_n004.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()

    print(f"Nyquist plot saved as:\n{output_file}")


# =========================================================
# 11. BODE MAGNITUDE PLOT
# =========================================================


def plot_bode_magnitude(simulated_z):

    magnitude = np.abs(simulated_z)

    plt.figure(figsize=(9, 6))

    smooth_f, smooth_mag = smooth_curve(magnitude)

    plt.semilogx(smooth_f, smooth_mag, color="tab:blue", linewidth=1.5)

    plt.semilogx(frequency, magnitude, linestyle="none", marker="o", color="tab:blue")

    for f, value in zip(frequency, magnitude):

        plt.annotate(
            make_label(f), (f, value), xytext=(7, 7), textcoords="offset points", fontsize=9
        )

    plt.xlabel("Frequency [Hz]")

    plt.ylabel("|Z| [Ω]")

    plt.title("Bode Magnitude Plot – LTspice Simulation")

    plt.grid(True, which="both")

    plt.tight_layout()

    output_file = FOLDER / "bode_magnitude_simulation_n004.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# =========================================================
# 12. BODE PHASE PLOT
# =========================================================


def plot_bode_phase(simulated_z):

    phase = np.degrees(np.angle(simulated_z))

    plt.figure(figsize=(9, 6))

    smooth_f, smooth_phase = smooth_curve(phase)

    plt.semilogx(smooth_f, smooth_phase, color="tab:blue", linewidth=1.5)

    plt.semilogx(frequency, phase, linestyle="none", marker="o", color="tab:blue")

    for f, value in zip(frequency, phase):

        plt.annotate(
            make_label(f), (f, value), xytext=(7, 7), textcoords="offset points", fontsize=9
        )

    plt.xlabel("Frequency [Hz]")

    plt.ylabel("Phase [degree]")

    plt.title("Bode Phase Plot – LTspice Simulation")

    plt.grid(True, which="both")

    plt.tight_layout()

    output_file = FOLDER / "bode_phase_simulation_n004.png"

    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    plt.show()


# =========================================================
# 13. MAIN PROGRAM
# =========================================================


def main():

    fft_frequencies, voltages, currents = read_ltspice_fft(INPUT_FILE)

    simulated_z = calculate_simulated_impedance(fft_frequencies, voltages, currents)

    theoretical_z = calculate_randles_impedance(frequency)

    print_results(simulated_z, theoretical_z)

    save_results(simulated_z, theoretical_z)

    plot_nyquist(simulated_z, theoretical_z)

    plot_bode_magnitude(simulated_z)

    plot_bode_phase(simulated_z)


# =========================================================
# RUN PROGRAM
# =========================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as error:

        print("\nError:")

        print(error)
