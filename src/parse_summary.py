import re

def parse_summary(summary_path):
    """
    Parses a CHB-MIT summary.txt file.
    Returns a dict: { filename: [(seizure_start, seizure_end), ...] }
    """
    with open(summary_path, "r") as f:
        content = f.read()

    # Split into blocks per file using "File Name:" as the separator
    blocks = content.split("File Name:")[1:]  # skip header before first file

    file_seizures = {}

    for block in blocks:
        filename_match = re.search(r"^\s*(\S+\.edf)", block)
        if not filename_match:
            continue
        filename = filename_match.group(1)

        # Two summary formats exist: "Seizure Start Time:" (single seizure) and
        # "Seizure 1 Start Time:" (numbered, for files with multiple seizures).
        starts = re.findall(r"Seizure (?:\d+ )?Start Time:\s*(\d+)\s*seconds", block)
        ends = re.findall(r"Seizure (?:\d+ )?End Time:\s*(\d+)\s*seconds", block)

        seizures = [(int(s), int(e)) for s, e in zip(starts, ends)]
        file_seizures[filename] = seizures

    return file_seizures


if __name__ == "__main__":
    result = parse_summary("data/raw/chb01/chb01-summary.txt")
    for filename, seizures in result.items():
        print(filename, "->", seizures)