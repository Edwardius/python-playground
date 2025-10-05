#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt, savgol_filter
import sys
import os
import argparse
import math

def load_data(filename, start_ms, end_ms):
    times = []
    angles = []
    controls = []

    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if line.strip():
                parts = line.strip().split(',')

                # Skip lines that don't have at least 2 values (time and angle)
                if len(parts) < 2:
                    continue

                try:
                    time_ms = float(parts[0])
                    angle_mrad = float(parts[1])

                    if start_ms <= time_ms <= end_ms:
                        control_mrad = float(parts[2]) if len(parts) > 2 else 0.0

                        times.append(time_ms)
                        angles.append(angle_mrad)
                        controls.append(control_mrad)
                except ValueError:
                    # Skip lines with invalid numeric data
                    print(f"Warning: Skipping line {line_num} with invalid data: {line.strip()}")
                    continue

    return np.array(times), np.array(angles), np.array(controls)

def find_control_edges(times, controls):
    """Find where control input changes"""
    control_diff = np.diff(controls)
    rising_edges = []
    falling_edges = []
    
    for i in range(len(control_diff)):
        if control_diff[i] > 0:  # Rising edge
            rising_edges.append(i + 1)
        elif control_diff[i] < 0:  # Falling edge
            falling_edges.append(i + 1)
    
    return rising_edges, falling_edges

def calculate_system_parameters(overshoot_percent, time_to_peak_s, Kp):
    """Calculate damping ratio and natural frequency from step response
    For system P(s) = K/(s(tau*s + 1)) with proportional controller Kp
    Closed-loop transfer function: KKp/(tau*s^2 + s + KKp)
    """
    if overshoot_percent <= 0 or time_to_peak_s <= 0:
        return None, None, None, None
    
    # Calculate damping ratio from percentage overshoot
    # %OS = exp(-zeta*pi/sqrt(1-zeta^2)) * 100
    ln_os = math.log(overshoot_percent / 100)
    zeta_squared = ln_os**2 / (math.pi**2 + ln_os**2)
    zeta = math.sqrt(zeta_squared)
    
    # Calculate natural frequency from time to peak
    # tp = pi / (wn * sqrt(1 - zeta^2))
    wn = math.pi / (time_to_peak_s * math.sqrt(1 - zeta**2))
    
    # For system P(s) = K/(s(tau*s + 1)) with proportional gain Kp:
    # Open-loop: KKp/(s(tau*s + 1))
    # Closed-loop: KKp/(tau*s^2 + s + KKp)
    # Standard form: wn^2/(s^2 + 2*zeta*wn*s + wn^2)
    
    # Comparing coefficients:
    # wn^2 = KKp/tau
    # 2*zeta*wn = 1/tau
    
    # From second equation: tau = 1/(2*zeta*wn)
    tau = 1 / (2 * zeta * wn)
    
    # From first equation: K = wn^2 * tau / Kp
    K = wn**2 * tau / Kp
    
    return zeta, wn, tau, K

def analyze_step_response(times, angles, controls, edge_idx, Kp):
    """Analyze the response after a control input change"""
    if edge_idx >= len(times) - 1:
        return None, None, None, None, None, None, None, None
    
    # Get the control target value after the edge
    control_target = controls[edge_idx]
    
    # Find the highest point in motor response after the edge
    angles_after = angles[edge_idx:]
    times_after = times[edge_idx:]
    
    if len(angles_after) == 0:
        return None, None, None, None, None, None, None, None
    
    max_idx = np.argmax(angles_after)
    max_angle = angles_after[max_idx]
    max_time = times_after[max_idx]
    
    # Calculate overshoot
    overshoot = max_angle - control_target
    
    # Calculate percentage overshoot
    if control_target != 0:
        overshoot_percent = (overshoot / control_target) * 100
    else:
        overshoot_percent = 0
    
    # Time to reach maximum (from control change)
    time_to_max = max_time - times[edge_idx]
    time_to_peak_s = time_to_max / 1000.0  # Convert to seconds
    
    # Calculate system parameters
    zeta, wn, tau, K = calculate_system_parameters(overshoot_percent, time_to_peak_s, Kp)
    
    return max_angle, time_to_max, overshoot, overshoot_percent, zeta, wn, tau, K

def create_interactive_plot(times, data, title, ylabel='Angle (mrad)', markers=None):
    """Create an interactive plot similar to plot_angle_data.py"""
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    # Convert to seconds for display
    times_s = times / 1000.0
    
    line, = ax.plot(times_s, data, 'b-', linewidth=2, label='Motor data')
    
    if markers is not None:
        for marker in markers:
            ax.axvline(x=marker['time']/1000.0, color=marker.get('color', 'r'), 
                      linestyle=marker.get('linestyle', '--'), 
                      label=marker.get('label', ''), alpha=0.7)
            if 'point' in marker:
                ax.plot(marker['time']/1000.0, marker['point'], 'ro', markersize=8)
                ax.annotate(f"{marker['annotation']}", 
                           xy=(marker['time']/1000.0, marker['point']),
                           xytext=(20, -40), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add zoom functionality
    def on_scroll(event):
        if event.inaxes == ax:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            
            xdata_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
            ydata_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
            
            scale_factor = 0.9 if event.button == 'up' else 1.1
            
            new_xlim = [xdata_center + (x - xdata_center) * scale_factor for x in xlim]
            new_ylim = [ydata_center + (y - ydata_center) * scale_factor for y in ylim]
            
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Analyze motor response to control input')
    parser.add_argument('filename', help='Input data file')
    parser.add_argument('start', type=float, help='Start timestamp (ms)')
    parser.add_argument('end', type=float, help='End timestamp (ms)')
    parser.add_argument('Kp', type=float, help='Proportional gain of the controller')
    parser.add_argument('--debug', action='store_true', help='Show debug plots')
    
    args = parser.parse_args()
    
    # Load data
    times, angles, controls = load_data(args.filename, args.start, args.end)
    
    if len(times) == 0:
        print("No data found in the specified time range")
        sys.exit(1)
    
    # Apply rolling median filter (window size 3 for minimal filtering)
    median_window = 3
    angles_median = medfilt(angles, median_window)

    # Apply moving average (Savitzky-Golay filter for smoothing)
    # Using smaller window for less aggressive smoothing
    window_length = min(51, len(angles_median) if len(angles_median) % 2 == 1 else len(angles_median) - 1)
    angles_smooth = savgol_filter(angles_median, window_length=window_length, polyorder=3)
    
    # Find control edges
    rising_edges, falling_edges = find_control_edges(times, controls)
    
    print(f"\nAnalysis for time range {args.start:.1f}ms to {args.end:.1f}ms")
    print(f"Found {len(rising_edges)} rising edges and {len(falling_edges)} falling edges")
    
    # Analyze each rising edge
    for i, edge_idx in enumerate(rising_edges):
        max_angle, time_to_max, overshoot, overshoot_percent, zeta, wn, tau, K = analyze_step_response(times, angles_smooth, controls, edge_idx, args.Kp)
        
        if max_angle is not None:
            print(f"\nRising edge {i+1} at {times[edge_idx]:.1f}ms:")
            print(f"  Control target: {controls[edge_idx]:.2f} mrad ({controls[edge_idx]/1000*180/np.pi:.2f}°)")
            print(f"  Maximum response: {max_angle:.2f} mrad ({max_angle/1000*180/np.pi:.2f}°)")
            print(f"  Overshoot: {overshoot:.2f} mrad ({overshoot/1000*180/np.pi:.2f}°)")
            print(f"  Percentage overshoot: {overshoot_percent:.1f}%")
            print(f"  Time to maximum: {time_to_max:.1f} ms")
            
            if zeta is not None:
                print(f"\nSystem parameters (with Kp = {args.Kp}):")
                print(f"  Damping ratio (ζ): {zeta:.3f}")
                print(f"  Natural frequency (ωn): {wn:.2f} rad/s")
                print(f"  Time constant (τ): {tau:.3f} s")
                print(f"  System gain (K): {K:.3f}")
                print(f"  Transfer function: P(s) = {K:.3f} / (s({tau:.3f}s + 1))")
    
    if args.debug:
        # Create a single comprehensive plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(left=0.1, bottom=0.25)
        
        times_s = times / 1000.0
        
        # Plot all processing stages
        ax.plot(times_s, angles, 'o', color='gray', markersize=3, label='Raw data', alpha=0.3)
        ax.plot(times_s, angles_median, 'cyan', linewidth=1.5, label='After median filter', alpha=0.5)
        ax.plot(times_s, angles_smooth, 'b-', linewidth=2, label='After moving average')
        ax.plot(times_s, controls, 'r--', linewidth=2, label='Control input', alpha=0.7)
        
        # Mark control changes and maximum responses
        for i, edge_idx in enumerate(rising_edges):
            # Mark control change
            ax.axvline(x=times[edge_idx]/1000.0, color='g', linestyle='--', 
                      label=f'Control change {i+1}' if i == 0 else '', alpha=0.7)
            
            # Mark maximum response
            max_angle, time_to_max, overshoot, overshoot_percent, zeta, wn, tau, K = analyze_step_response(times, angles_smooth, controls, edge_idx, args.Kp)
            if max_angle is not None:
                max_time = times[edge_idx] + time_to_max
                ax.plot(max_time/1000.0, max_angle, 'ro', markersize=8)
                ax.annotate(f"Max: {max_angle:.1f} mrad\nTime: {time_to_max:.1f} ms\nOvershoot: {overshoot_percent:.1f}%", 
                           xy=(max_time/1000.0, max_angle),
                           xytext=(20, -40), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3'))
        
        # Add system parameters text box
        if rising_edges:
            # Use parameters from first rising edge
            edge_idx = rising_edges[0]
            max_angle, time_to_max, overshoot, overshoot_percent, zeta, wn, tau, K = analyze_step_response(times, angles_smooth, controls, edge_idx, args.Kp)
            
            if zeta is not None:
                param_text = f"System Parameters (Kp = {args.Kp})\n"
                param_text += f"Transfer Function: P(s) = {K:.3f} / (s({tau:.3f}s + 1))\n"
                param_text += f"Damping Ratio (ζ) = {zeta:.3f}\n"
                param_text += f"Natural Frequency (ωn) = {wn:.2f} rad/s"
                
                ax.text(0.98, 0.02, param_text, transform=ax.transAxes, 
                       verticalalignment='bottom', horizontalalignment='right', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Angle (milliradians)', fontsize=12)
        ax.set_title('Motor Response Analysis - All Processing Stages', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add zoom functionality
        def on_scroll(event):
            if event.inaxes == ax:
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                
                xdata_center = event.xdata if event.xdata else (xlim[0] + xlim[1]) / 2
                ydata_center = event.ydata if event.ydata else (ylim[0] + ylim[1]) / 2
                
                scale_factor = 0.9 if event.button == 'up' else 1.1
                
                new_xlim = [xdata_center + (x - xdata_center) * scale_factor for x in xlim]
                new_ylim = [ydata_center + (y - ydata_center) * scale_factor for y in ylim]
                
                ax.set_xlim(new_xlim)
                ax.set_ylim(new_ylim)
                fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect('scroll_event', on_scroll)

        # Save the figure instead of showing it
        output_filename = f"motor_response_{args.start}_{args.end}_ms.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {output_filename}")

if __name__ == "__main__":
    main()