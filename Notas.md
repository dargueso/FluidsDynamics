## Notes to install dedalus 2

    conda create -n dedalus2 python=3.10
    conda activate dedalus2
    conda search -f dedalus
    conda install dedalus=2.2207.3 #Depending on the versions shown with the previous command

Then run the python script with dedalus

    export OMP_NUM_THREADS=1
    python Rayleigh-Taylor_instability_viscous_diffusive.py 

## Notes to run a python script using dedalus in megacelula3

Access megacelula3 with ssh using usearname grupo1

The run the following commands to set the environment. 

    bash
    unset PYTHONPATH
    cd /home/grupo1/Master_FAMA_Inestabilitats/
    conda activate dedalus2
    export OMP_NUM_THREADS=1

    python Rayleigh-Taylor_instability_viscous_diffusive.py 
