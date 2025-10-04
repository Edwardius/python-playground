#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
import sys
import os

def process_and_plot(filename):
    times = []
    angles = []
    controls = []
    has_control = False
    
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split(',')
                time_ms = float(parts[0])
                angle_mrad = float(parts[1])
                
                time_s = time_ms / 1000.0
                angle_rad = angle_mrad / 1000.0
                
                times.append(time_s)
                angles.append(angle_rad)
                
                if len(parts) > 2:
                    has_control = True
                    control_mrad = float(parts[2])
                    control_rad = control_mrad / 1000.0
                    controls.append(control_rad)
    
    times = np.array(times)
    angles = np.array(angles)
    if has_control:
        controls = np.array(controls)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    
    line, = ax.plot(times, angles, 'b-', linewidth=2, label='Angle data', picker=True, pickradius=5)
    
    if has_control:
        control_line, = ax.plot(times, controls, 'r--', linewidth=2, label='Control input', alpha=0.7)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Angle (radians)', fontsize=12)
    ax.set_title(f'Angle vs Time - {os.path.basename(filename)}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.03])
    time_slider = Slider(ax_slider, 'Time Window', 0, times[-1] - times[0], 
                         valinit=times[-1] - times[0], valstep=0.1)
    
    ax_reset = plt.axes([0.8, 0.025, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset', hovercolor='0.975')
    
    initial_xlim = ax.get_xlim()
    initial_ylim = ax.get_ylim()
    
    def update_view(val):
        window_size = time_slider.val
        if window_size < 0.1:
            window_size = 0.1
        
        xlim = ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        
        new_xmin = center - window_size / 2
        new_xmax = center + window_size / 2
        
        if new_xmin < times[0]:
            new_xmin = times[0]
            new_xmax = times[0] + window_size
        elif new_xmax > times[-1]:
            new_xmax = times[-1]
            new_xmin = times[-1] - window_size
        
        ax.set_xlim(new_xmin, new_xmax)
        
        visible_mask = (times >= new_xmin) & (times <= new_xmax)
        if visible_mask.any():
            visible_angles = angles[visible_mask]
            y_margin = (visible_angles.max() - visible_angles.min()) * 0.1
            ax.set_ylim(visible_angles.min() - y_margin, visible_angles.max() + y_margin)
        
        fig.canvas.draw_idle()
    
    def reset(event):
        time_slider.reset()
        ax.set_xlim(initial_xlim)
        ax.set_ylim(initial_ylim)
        fig.canvas.draw_idle()
    
    def onselect(xmin, xmax):
        ax.set_xlim(xmin, xmax)
        
        visible_mask = (times >= xmin) & (times <= xmax)
        if visible_mask.any():
            visible_angles = angles[visible_mask]
            y_margin = (visible_angles.max() - visible_angles.min()) * 0.1
            ax.set_ylim(visible_angles.min() - y_margin, visible_angles.max() + y_margin)
        
        time_slider.set_val(xmax - xmin)
        fig.canvas.draw_idle()
    
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True,
                       props=dict(alpha=0.3, facecolor='yellow'))
    
    time_slider.on_changed(update_view)
    button_reset.on_clicked(reset)
    
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
            
            time_slider.set_val(new_xlim[1] - new_xlim[0])
            fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    # Add annotation for showing point values
    annot = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="yellow", alpha=0.9),
                        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.3"))
    annot.set_visible(False)
    
    def update_annot(x, y, label="Angle"):
        annot.xy = (x, y)
        text = f"Time: {x:.3f}s\n{label}: {y:.6f} rad"
        annot.set_text(text)
        color = "yellow" if label == "Angle" else "lightcoral"
        annot.get_bbox_patch().set_facecolor(color)
        annot.get_bbox_patch().set_alpha(0.9)
    
    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            # Check angle line
            cont_angle, ind_angle = line.contains(event)
            cont_control = False
            if has_control:
                cont_control, ind_control = control_line.contains(event)
            
            if cont_angle or cont_control:
                mouse_x = event.xdata
                mouse_y = event.ydata
                
                # Find index of closest point
                if cont_angle and (not cont_control):
                    distances = np.sqrt((times - mouse_x)**2 + ((angles - mouse_y) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))**2)
                    closest_idx = np.argmin(distances)
                    update_annot(times[closest_idx], angles[closest_idx], "Angle")
                elif cont_control and (not cont_angle):
                    distances = np.sqrt((times - mouse_x)**2 + ((controls - mouse_y) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))**2)
                    closest_idx = np.argmin(distances)
                    update_annot(times[closest_idx], controls[closest_idx], "Control")
                else:
                    # Both lines are close, choose the closer one
                    dist_angle = np.sqrt((times - mouse_x)**2 + ((angles - mouse_y) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))**2)
                    dist_control = np.sqrt((times - mouse_x)**2 + ((controls - mouse_y) * (ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]))**2)
                    idx_angle = np.argmin(dist_angle)
                    idx_control = np.argmin(dist_control)
                    
                    if dist_angle[idx_angle] < dist_control[idx_control]:
                        update_annot(times[idx_angle], angles[idx_angle], "Angle")
                    else:
                        update_annot(times[idx_control], controls[idx_control], "Control")
                
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()
    
    fig.canvas.mpl_connect("motion_notify_event", hover)
    
    print(f"Processed {len(times)} data points")
    print(f"Time range: {times.min():.3f}s to {times.max():.3f}s")
    print(f"Angle range: {angles.min():.6f} to {angles.max():.6f} radians")
    if has_control:
        print(f"Control range: {controls.min():.6f} to {controls.max():.6f} radians")
    print("\nInteractive controls:")
    print("- Hover over points to see their values")
    print("- Click and drag to select a time range to zoom")
    print("- Use mouse wheel to zoom in/out")
    print("- Use the slider to adjust the time window size")
    print("- Click Reset to return to full view")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_angle_data.py <input_file.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    process_and_plot(input_file)