import matplotlib.pyplot as plt
import numpy as np
import re

def main():
    print("NumPy version:", np.__version__)
    data = read_can_data("data/RenaultClio/dosattack.log")
    print(data[:5])

    timestamps = data[:, 0].astype(float)
    priorities = data[:, 1].astype(int)
    
    time_intervals = np.diff(timestamps)
    plot_histograms(time_intervals, priorities)

def plot_histograms(time_intervals, priorities):
    plt.figure(figsize=(12, 5))
    
    # Histogram for time intervals
    plt.subplot(1, 2, 1)
    plt.hist(time_intervals, bins=50, color='b', alpha=0.7)
    plt.xlabel("Time Between Packets (s)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Time Intervals")
    
    # Histogram for priorities
    plt.subplot(1, 2, 2)
    plt.hist(priorities, bins=50, color='r', alpha=0.7)
    plt.xlabel("Packet Priority (Hex)")
    plt.ylabel("Frequency")
    plt.title("Histogram of Packet Priorities")
    
    plt.tight_layout()
    plt.show()

def read_can_data(filepath):
    data = []

    with open(filepath, 'r') as file:
        for line in file:
            match = re.match(r"\((\d+\.\d+)\)\s+\w+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)", line)
            if match:
                time = float(match.group(1))
                priority = int(match.group(2), 16)  # Convert hex to int
                payload = match.group(3)  # Keep data as hex string
                
                data.append([time, priority, payload])
    
    return np.array(data, dtype=object)


if __name__ == "__main__":
    main()