#!/bin/bash
python NuGen.py -n 1000000  -e 3:7 -T NuTau:NuTauBar -s Full -D /data/user/yxu/icesim_V050101/src/neutrino-generator/cross_section_data -x csms
python NuGen.py -n 1000000  -e 3:7 -T NuTau:NuTauBar -s Full -D /data/user/yxu/icesim_V050101/src/neutrino-generator/cross_section_data -x csms_02
python NuGen.py -n 1000000  -e 3:7 -T NuTau:NuTauBar -s Full -D /data/user/yxu/icesim_V050101/src/neutrino-generator/cross_section_data -x csms_500
