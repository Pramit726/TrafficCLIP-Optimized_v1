import logging
from pathlib import Path

import numpy as np
import scapy.all as scapy
import scipy

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


def calculate_stats(flow_pkts, raw_bytes):
    """
    Calculates physical flow characteristics: Mean IAT, Jitter, and Byte Entropy.
    Calculated before truncation/masking for maximum accuracy.
    """
    # 1. Calculate Inter-Arrival Times (IAT)
    timestamps = [float(p.time) for p in flow_pkts]
    if len(timestamps) > 1:
        iats = np.diff(timestamps)
        # Convert to milliseconds for the prompt
        mean_iat = np.mean(iats) * 1000
        jitter = np.std(iats) * 1000
    else:
        mean_iat, jitter = 0.0, 0.0

    # 2. Calculate Byte Entropy (Shannon Entropy)
    if len(raw_bytes) > 0:
        counts = np.bincount(np.frombuffer(raw_bytes, dtype=np.uint8), minlength=256)
        probs = counts / len(raw_bytes)
        entropy = scipy.stats.entropy(probs, base=2)
    else:
        entropy = 0.0

    return mean_iat, jitter, entropy


def flow_to_image_and_stats(flow_packets, size=784):
    """
    Extracts stats and converts flow to a 54-byte masked image.
    """
    # Step A: Get FULL raw bytes for stats
    raw_bytes = b"".join([scapy.raw(p) for p in flow_packets])

    # Step B: Physics-Informed Calculation
    mean_iat, jitter, entropy = calculate_stats(flow_packets, raw_bytes)

    # Step C: Prepare Image Buffer (Truncate/Pad)
    buffer = bytearray(raw_bytes[:size])
    if len(buffer) < size:
        buffer.extend(b"\x00" * (size - len(buffer)))

    # Step D: 54-Byte Header Masking (Phase 1 Fix)
    # Masking Eth(14) + IP(20) + TCP(20) to prevent hardware fingerprints
    for i in range(0, 54):
        buffer[i] = 0x00

    img = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32) / 255.0
    return img.reshape(1, 28, 28), [mean_iat, jitter, entropy]


def process_class_folder(folder: Path, label: int, samples_per_class: int):
    images, labels, metadata = [], [], []
    count = 0
    logging.info(f"Processing Class: {folder.name}")

    for pcap_file in folder.glob("*.pcap"):
        if count >= samples_per_class:
            break
        try:
            packets = scapy.rdpcap(str(pcap_file))
        except:
            continue

        flows = extract_flows(packets)
        for flow_packets in flows.values():
            if count >= samples_per_class:
                break

            img, stats = flow_to_image_and_stats(flow_packets)
            images.append(img)
            labels.append(label)
            metadata.append(stats)
            count += 1

    return images, labels, metadata


def process_pcaps_to_numpy(data_dir: Path, output_file: Path, samples_per_class=1500):
    all_images, all_labels, all_metadata = [], [], []
    class_folders, label_map = get_class_folders(data_dir)

    for folder in class_folders:
        label = label_map[folder.name]
        imgs, lbls, meta = process_class_folder(folder, label, samples_per_class)
        all_images.extend(imgs)
        all_labels.extend(lbls)
        all_metadata.extend(meta)

    # Save with 'm' key for Phase 2 Dynamic Prompting
    np.savez_compressed(
        output_file,
        x=np.array(all_images),
        y=np.array(all_labels),
        m=np.array(all_metadata),
        labels=list(label_map.keys()),
    )

    logging.info(f"Successfully saved {len(all_images)} samples to {output_file}")


if __name__ == "__main__":
    config = load_config()

    RAW_DATA_PATH = Path(config["paths"]["raw_data_dir"])
    OUTPUT_FILE = Path(config["paths"]["output_data_file"])
    MINI_OUTPUT_FILE = Path(config["paths"]["mini_output_data_file"])

    # process_pcaps_to_numpy(RAW_DATA_PATH, OUTPUT_FILE, samples_per_class=900)
    process_pcaps_to_numpy(RAW_DATA_PATH, MINI_OUTPUT_FILE, samples_per_class=10)
