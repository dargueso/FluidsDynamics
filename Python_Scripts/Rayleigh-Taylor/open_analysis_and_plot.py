#!/usr/bin/env python
'''
@File    :  open_analysis_and_plot.py
@Time    :  2023/02/13 18:58:24
@Author  :  Daniel Argüeso
@Version :  1.0
@Contact :  d.argueso@uib.es
@License :  (C)Copyright 2023, Daniel Argüeso
@Project :  Geophysical Fluid Dynamics - Instabilities
@Desc    :  None
'''

import h5py
import matplotlib.pyplot as plt
import numpy as np


with h5py.File("analysis/analysis_s1/analysis_s1_p0.h5", mode="r") as file:
    rho = file["tasks"]["rho"]
    t = rho.dims[0]["sim_time"]
    x = rho.dims[1][0]
    y = rho.dims[2][0]

    for tstep in range(len(t)):
        plt.figure(figsize=(15, 5), dpi=100)
        ct = plt.pcolormesh(
            x[:],
            y[:],
            rho[tstep, :, :].T,
            cmap='RdYlBu'
        )
        plt.title('Density')
        plt.xlim([0,2.])
        plt.ylim([-0.5,0.5])
        plt.tight_layout()
        plt.savefig(f"R-T_{tstep:03d}.png")
        plt.close()



