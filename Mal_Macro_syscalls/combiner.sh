#!/bin/bash

# Combine zpoline_B4_syscalls#.csv with zpoline_syscalls#.csv
# Get rid of first line in zpoline_syscalls#.csv
# Put second line of zpoline_syscalls#.csv in timestamps#.csv

for i in {1..10}; 
do
  before_file="zpoline_B4_syscalls$i.csv"
  after_file="zpoline_syscalls$i.csv"
  timestamp_file="timestamps$i.csv"
  combined_file="combined_syscalls$i.csv"

  if [[ -f "$before_file" && -f "$after_file" ]]; then
    
    sed -n '2p' "$after_file" > "$timestamp_file"

    tail -n +2 "$after_file" > "trimmed_$i.csv"

    cat "$before_file" "trimmed_$i.csv" > "$combined_file"

    rm "trimmed_$i.csv"

    echo "Files $i were combined"
  else
    echo "Files $i were not found."
  fi
done
