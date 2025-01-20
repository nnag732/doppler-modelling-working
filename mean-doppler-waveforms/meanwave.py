import os
import numpy as np
import placentagen as pg
import csv
from matplotlib import pyplot as plt
import cv2
import matplotlib.image as mpimg
#try delete the 3 scipy
from scipy import interpolate
from scipy.optimize import curve_fit, minimize,least_squares
from scipy.interpolate import  UnivariateSpline, Akima1DInterpolator, PchipInterpolator, interp1d
from scipy.signal import find_peaks

from reprosim.diagnostics import set_diagnostics_level
from reprosim.indices import perfusion_indices, get_ne_radius
from reprosim.geometry import append_units,define_node_geometry, define_1d_element_geometry,define_1d_element_placenta,define_rad_from_geom,add_matching_mesh, \
        define_capillary_model,define_rad_from_file
from reprosim.repro_exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field,export_node_field,export_terminal_perfusion
from reprosim.fetal import assign_fetal_arrays, fetal_model


#inlet_rad = 1.5
dt=0.001 #s, time step
weight_38w = 3.4 #kg Not used, but current parameterisation assumes an average fetal weight and allometric scaling may be useful in future
weight_new = 2.958

def read_xy_coords():  
    x_values = []
    y_values = []

    # Read CSV file
    with open('_DAPHNE-46_20230612_bae9d604.csv', 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            values = line.strip().split(',')
            x = float(values[0])  
            y = float(values[1])  
            x_values.append(x)
            y_values.append(y)

    # Convert lists to numpy arrays
    x_values_np = np.array(x_values)
    y_values_np = np.array(y_values)

    # Find troughs (local minima) in y_values
    trough_indices, _ = find_peaks(-y_values_np)  # Negate y_values to find minima

    #plot troughs
    plt.figure(figsize=(10, 6))
    plt.plot(x_values_np, y_values_np, label="Waveform", color='blue')
    plt.scatter(x_values_np[trough_indices], y_values_np[trough_indices], color='red', label='Troughs', zorder=5)
    plt.title("Waveform with Trough Points")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    ################  average across multiple waves  #############################
    interpolated_waves = []

    for i in range(len(trough_indices) - 1):
        # Extract data for the current segment
        start_index = trough_indices[i]
        end_index = trough_indices[i + 1]
        x_segment = x_values_np[start_index:end_index]
        y_segment = y_values_np[start_index:end_index]
        
        # Shift x-coordinates for alignment (except the first wave)
        if i > 0:
            x_segment = x_segment - (x_segment[0] - x_values_np[trough_indices[0]])

        # Initialize the common x-axis using the first segment
        if i == 0:
            x_min = x_segment[0]
            x_max = x_segment[-1]
            x_common_points = len(x_segment)
            x_common = np.linspace(x_min, x_max, x_common_points)

        # Interpolate to the common x-axis
        interp_y = interp1d(x_segment, y_segment, kind='linear', fill_value="extrapolate")(x_common)
        interpolated_waves.append(interp_y)

    # Convert the list of interpolated waves to a NumPy array for calculations
    interpolated_waves_np = np.vstack(interpolated_waves)

    # Calculate the initial average and standard deviation
    average_wave = np.mean(interpolated_waves_np, axis=0)
    std_wave = np.std(interpolated_waves_np, axis=0)

    # Filter out waves outside the range of average ± standard deviation
    threshold_percentage = 80
    
    filtered_waves = []
    for wave in interpolated_waves_np:
        # Calculate the percentage of points that meet the OR condition
        within_range = (wave >= (average_wave - std_wave)) &  (wave <= (average_wave + std_wave)) # Points above or equal to lower bound
        percentage_within_range = np.sum(within_range) / len(wave) * 100
        
        # Include the wave if the percentage is above the threshold
        if percentage_within_range >= threshold_percentage:
            filtered_waves.append(wave)

    # Recalculate the average and standard deviation with the filtered waves
    filtered_waves_np = np.vstack(filtered_waves)
    new_average_wave = np.mean(filtered_waves_np, axis=0)
    new_std_wave = np.std(filtered_waves_np, axis=0)

    # average_wave = new_average_wave
    # std_wave = new_std_wave

    #just the first 2 waves###################################################
    # Extract the data between the first two troughs
    # start_index = trough_indices[0]
    # end_index = trough_indices[1]
    # x1 = x_values_np[start_index:end_index]
    # y1 = y_values_np[start_index:end_index]

    # start_index = trough_indices[1]
    # end_index = trough_indices[2]
    # x2 = x_values_np[start_index:end_index]
    # #shift x2 coords
    # x2=x2-(x2[0]-x1[0])
    # y2 = y_values_np[start_index:end_index]

    # # Determine the average x-axis range and spacing
    # x_min = min(x1[0], x2[0])  # Minimum of both x ranges
    # x_max = max(x1[-1], x2[-1])  # Maximum of both x ranges
    
    # x_common_points = (len(x1) + len(x2)) // 2  # Average number of points
    # x_common = np.linspace(x_min, x_max, x_common_points)  # Common x-axis

    # # Interpolate both waves to the common x-axis
    # interp_y1 = interp1d(x1, y1, kind='linear', fill_value="extrapolate")(x_common)
    # interp_y2 = interp1d(x2, y2, kind='linear', fill_value="extrapolate")(x_common)

    # # Calculate the average and standard deviation
    # average_wave = (interp_y1 + interp_y2) / 2
    # std_wave = np.std(np.vstack([interp_y1, interp_y2]), axis=0)
    #############################################################################

    # Plot all waves and the average wave for visualization
    plt.figure(figsize=(12, 8))
    for i, interp_y in enumerate(interpolated_waves):
        plt.plot(x_common, interp_y, label=f'Wave {i+1} (Original)', alpha=0.6, linestyle='--')
    plt.plot(x_common,  new_average_wave,'r-', label='Filtered Average Wave', linewidth=2)
    plt.fill_between(x_common, new_average_wave - new_std_wave, new_average_wave + new_std_wave,
                    color='r', alpha=0.2, label='±1 filtered STD')
    # plt.plot(x_common, average_wave, 'k-', label='Average Wave', linewidth=2)
    # plt.fill_between(x_common, average_wave - std_wave, average_wave + std_wave,
    #                 color='k', alpha=0.2, label='±1 STD')
    
    plt.legend()
    plt.xlabel('X-axis')
    plt.ylabel('Amplitude')
    plt.title('Original Waves and Filtered average')
    plt.grid()
    plt.show()

    
    # # Plot the original and averaged waves for visualization
    # plt.figure(figsize=(10, 6))
    # plt.plot(x1, y1, 'o-', label='Wave 1 (Original)', alpha=0.7)
    # plt.plot(x2, y2, 's-', label='Wave 2 (Original)', alpha=0.7)
    # plt.plot(x_common, average_wave, 'r-', label='Average Wave', linewidth=2)
    # plt.fill_between(x_common, average_wave - std_wave, average_wave + std_wave,
    #              color='r', alpha=0.2, label='±1 STD')
    # plt.legend()
    # plt.xlabel('X-axis')
    # plt.ylabel('Amplitude')
    # plt.title('Wave 1, Wave 2, and Average Wave')
    # plt.grid()
    # plt.show()

    x_values_subset=x_common
    y_values_subset=new_average_wave

    #save averaged waveform to a txt file
    output_data = np.column_stack((x_values_subset,y_values_subset))
    np.savetxt('DAPHNE-46_20230612_bae9d604_Rt_avgwave.txt', output_data, header='Time(s) Velocity(cm/s)', delimiter='\t')

    return x_values_subset, y_values_subset, std_wave

def process_xy_coords():
    x_coordinates, y_coordinates, std_wave = read_xy_coords()

    ## Define time points at which you want to plot flows
    dt=0.01 #time step for plotting

    ## Calculate total duration of the heart beat
    total_duration = max(x_coordinates)-min(x_coordinates) + dt #adding an extra time step to this to fix an error in interpolation, may need to fix later
    
    print('Total duration = ', total_duration, 's')
    heart_rate = 30/total_duration
    print('Estimated heart rate = ',  heart_rate, 'bpm')
    HeartRate = heart_rate
    #Number of flow harmonics

    NHar=10
    #First ten harmonics of incident waveform [[omega],[A_n],[Phi_n]]. See Mo et al. A transmission line modelling approach to the interpretation of uterine doppler waveforms. Ultrasound in Medicine and Biology, 1988. 14(5): p. 365-376
    IWavHar=np.array([[1.0*HeartRate/60.0, 2.0*HeartRate/60.0, 3.0*HeartRate/60.0, 4.0*HeartRate/60.0, 5.0*HeartRate/60.0, 6.0*HeartRate/60.0, 7.0*HeartRate/60.0, 8.0*HeartRate/60.0, 9.0*HeartRate/60.0, 10.0*HeartRate/60.0],
                    [64.32,  43.08,  21.48,   7.68,   2.64,  1.8,    0.96,   0.84,   0.84,   0.48],
                    [-1.375319451, -2.138028334,-2.998475655,-3.548254369,-3.394665395,-3.185225885,-3.131120678,-2.448696941,-2.602285915,-2.441715624]])
            
    
    StartTime=0.0
    EndTime=60.0/HeartRate #end of a single beat
    

    # Calculate the time offset to align the two traces
    time_offset = StartTime - min(x_coordinates)

    # Shift the x-coordinates and time to align the traces
    x_coordinates= [t + time_offset for t in x_coordinates] 

    # Sort the extracted data by "time" or x_coordinate
    x_coordinates = np.array(x_coordinates)
    y_coordinates = np.array(y_coordinates)
    sorted_idx = np.argsort(x_coordinates)
    return x_coordinates[sorted_idx], y_coordinates[sorted_idx], std_wave


def main():
    
    #Get the trace detail from image
    x_coordinates, y_coordinates, std_wave = process_xy_coords()
   
if __name__ == '__main__':
    main()
