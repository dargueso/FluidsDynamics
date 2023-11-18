# FluidsDynamics
 Scripts and materials for Waves and Instabilities in Geophysical Fluids course Master in Avanced Physics and Applied Mathematics



# Set up steps in our machines (Dpt. of Physics, Group of Meteorology)

```
ssh grupo1@eady.uib.es
ssh mc3
bash
unset PYTHONPATH
conda activate dedalus2
conda env config vars set OMP_NUM_THREADS=1
conda env config vars set NUMEXPR_MAX_THREADS=1
conda activate dedalus2

```

# Code to open dedalus analyses

Instead of making plots on the fly, we can generate analyses and then open them with a python script. 
There is an example in Rayleigh-Taylor [open_analysis_and_plot.py](./Jupyter_notebooks/Rayleigh-Taylor/open_analysis_and_plot.py)
