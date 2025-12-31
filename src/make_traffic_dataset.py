import logging
from pathlib import Path

import numpy as np
import scapy.all as scapy

from src.utils.utils import load_config


def get_class_folders(data_dir: Path) -> tuple[list[Path], dict[str, int]]:
    """
    Identifies class folders in the given directory and creates a label mapping.
    """
    class_folders = [f for f in data_dir.iterdir() if f.is_dir()]
    label_map = {folder.name: i for i, folder in enumerate(class_folders)}
    logging.info(f"Label Mapping: {label_map}")
    return class_folders, label_map


def read_pcap(pcap_file: Path) -> scapy.PacketList | None:
    """
    Reads a PCAP file and returns the packets.
    """
    try:
        return scapy.rdpcap(str(pcap_file))
    except Exception as e:
        logging.error(f"Failed to read {pcap_file.name}: {e}")
        return None


def extract_flows(packets) -> dict[tuple, list[scapy.Packet]]:
    """
    Extracts flows from a list of packets.
    """
    flows = {}

    for pkt in packets:
        if not pkt.haslayer(scapy.IP):
            continue

        ip = pkt[scapy.IP]
        proto = ip.proto
        sport, dport = 0, 0

        if pkt.haslayer(scapy.TCP):
            sport, dport = pkt[scapy.TCP].sport, pkt[scapy.TCP].dport
        elif pkt.haslayer(scapy.UDP):
            sport, dport = pkt[scapy.UDP].sport, pkt[scapy.UDP].dport

        flow_id = (
            tuple(sorted((ip.src, ip.dst))) + tuple(sorted((sport, dport))) + (proto,)
        )

        flows.setdefault(flow_id, []).append(pkt)

    return flows


def flow_to_image(flow_packets, size=784) -> np.ndarray:
    """
    Converts a flow's packets to a normalized 28x28 image array.
    """
    raw_bytes = b"".join([scapy.raw(p) for p in flow_packets])

    buffer = bytearray(raw_bytes[:size])
    if len(buffer) < size:
        buffer.extend(b"\x00" * (size - len(buffer)))

    # Anonymize IPv4 addresses (bytes 12–19)
    for i in range(12, 20):
        buffer[i] = 0x00

    img = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0

    return img.reshape(1, 28, 28)


def process_class_folder(
    folder: Path, label: int, samples_per_class: int
) -> tuple[list[np.ndarray], list[int]]:
    """
    Processes all PCAP files in a class folder and returns images and labels.
    """
    images, labels = [], []
    count = 0

    logging.info(f"Processing Class: {folder.name}")

    for pcap_file in folder.glob("*.pcap"):
        if count >= samples_per_class:
            break

        packets = read_pcap(pcap_file)
        if packets is None:
            continue

        flows = extract_flows(packets)

        for flow_packets in flows.values():
            if count >= samples_per_class:
                break

            img = flow_to_image(flow_packets)
            images.append(img)
            labels.append(label)
            count += 1

    return images, labels


def process_pcaps_to_numpy(
    data_dir: Path, output_file: Path, samples_per_class=1500
) -> None:
    """
    Reads PCAPs, extracts 784 bytes per flow, anonymizes IPs,
    and saves as a compressed NumPy archive (.npz).
    """
    all_images = []
    all_labels = []

    class_folders, label_map = get_class_folders(data_dir)

    for folder in class_folders:
        label = label_map[folder.name]
        imgs, lbls = process_class_folder(folder, label, samples_per_class)
        all_images.extend(imgs)
        all_labels.extend(lbls)

    x = np.array(all_images)
    y = np.array(all_labels)

    np.savez_compressed(output_file, x=x, y=y, labels=list(label_map.keys()))

    logging.info(f"Saved {x.shape[0]} samples")
    logging.info(f"Final shape: {x.shape} (N, C, H, W)")


if __name__ == "__main__":
    config = load_config()

    RAW_DATA_PATH = Path(config["paths"]["raw_data_dir"])
    OUTPUT_FILE = Path(config["paths"]["output_data_file"])

    process_pcaps_to_numpy(RAW_DATA_PATH, OUTPUT_FILE, samples_per_class=1500)
