from main import read_can_data
import matplotlib.pyplot as plt
def main():
    opeldata = read_can_data("data/OpelAstra/full_data_capture.log")
    print_stats(opeldata, "opelAstra")
    return


def print_stats(data, name: str):
    print(name)
    print(len(data))
    packet_lengths = [len(entry[2]) // 2 for entry in data]
    plt.figure(figsize=(8,5))
    plt.hist(packet_lengths, bins=range(1, 10), edgecolor='black', align='left', color='orange')
    plt.xticks(range(1, 9))
    plt.xlabel("Packet Length (DLC)")
    plt.ylabel("Frequency")
    plt.title("Histogram of CAN Packet Lengths")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()