#!/bin/bash

# Define input files
line_numbers_file="lines.txt"  # File containing line numbers (one per line)
input_file="hate-undesired.jsonl"  # Original text file
output_file="discard-hate-undesired.jsonl"  # File to store extracted lines
temp_file="temp.txt"  # Temporary file to hold non-matching lines

# Use awk to process lines
awk 'NR==FNR {lines[$1+1]; next} FNR in lines' "$line_numbers_file" "$input_file" > "$output_file"
awk 'NR==FNR {lines[$1+1]; next} !(FNR in lines)' "$line_numbers_file" "$input_file" > "$temp_file"

# Replace the original file with the updated content (without moved lines)
mv "$temp_file" "$input_file"

echo "Next lines removed and moved to $output_file. Updated $input_file."

