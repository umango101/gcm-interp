import json

def count_qc_failures(filepath):
    counts = {"qc1_ok": 0, "qc2_ok": 0, "qc3_ok": 0, "qc4_ok": 0}
    total = 0

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            total += 1
            for qc in counts:
                if not record.get(qc, True):
                    counts[qc] += 1

    print(f"Total lines: {total}")
    for qc, fail_count in counts.items():
        print(f"{qc} failures: {fail_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python qc_counter.py <path_to_jsonl>")
        sys.exit(1)
    count_qc_failures(sys.argv[1])
