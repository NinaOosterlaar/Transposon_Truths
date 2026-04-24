import subprocess

bam_file = "yEK23a_23_FDDP220013504-2a_HTWL7DSX2_L4_trimmed_forward_notrimmed.bam"
samtools_path = "/opt/homebrew/bin/samtools"  # adjust if needed

result = subprocess.run(
    [samtools_path, "stats", bam_file],
    capture_output=True,
    text=True,
    check=True
)

stats = {}

for line in result.stdout.splitlines():
    if line.startswith("SN"):
        parts = line.split("\t")
        if len(parts) >= 3:
            key = parts[1].rstrip(":")
            value = parts[2].strip()
            stats[key] = value

raw_total_sequences = int(stats["raw total sequences"])
average_length = float(stats["average length"])
max_length = int(stats["maximum length"])

print("Raw total sequences:", raw_total_sequences)
print("Average length:", average_length)
print("Maximum length:", max_length)