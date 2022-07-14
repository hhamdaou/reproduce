#!/bin/bash
python fullSimulation_csms1.py -n 1000 --seed=1 --datasetnumber=1 --runnumber=001 --detector=IC86 --flavor=NuTau --from-energy 1000 --to-energy 10000000 --outfile=NuTau_CSMS_1_2.i3.zst --no-auto-extend-muon-volume
