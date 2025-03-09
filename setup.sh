#!/bin/bash
sudo sh -c "echo 0 > /proc/sys/vm/mmap_min_addr"
LIBZPHOOK=/zpoline/apps/basic/libzphook_basic.so LD_PRELOAD=/zpoline/libzpoline.so libreoffice --impress
