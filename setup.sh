#!/bin/bash
sudo sh -c "echo 0 > /proc/sys/vm/mmap_min_addr"

read -p "Enter path to Impress file (or leave blank to open Impress without a file): " FILEPATH

# Remove surrounding quotes if present
FILEPATH="${FILEPATH%\"}"
FILEPATH="${FILEPATH#\"}"

if [ -n "$FILEPATH" ]; then
    if [ -f "$FILEPATH" ]; then
        LIBZPHOOK=/zpoline/apps/basic/libzphook_basic.so \
        LD_PRELOAD=/zpoline/libzpoline.so \
        libreoffice --impress "$FILEPATH"
    else
        echo "Error: File does not exist or is not a regular file."
        exit 1
    fi
else
    LIBZPHOOK=/zpoline/apps/basic/libzphook_basic.so \
    LD_PRELOAD=/zpoline/libzpoline.so \
    libreoffice --impress
fi
