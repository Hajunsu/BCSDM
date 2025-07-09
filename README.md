# BCSDM

## Requirements
### Environment
The project has been developed in a standard Anaconda environment with CUDA 12.6. To install all dependencies, simply run the following commands.

```shell
conda create -n bcsdm python=3.12
conda activate bcsdm
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip3 install pybullet
pip install tensorboardX wandb matplotlib scipy tqdm
pip install --upgrade omegaconf
```

## Download Additional Resources

This project requires additional resources to run or reproduce key results, including:

- datasets
- results
- assets

Due to file size constraints, these resources are provided separately from the GitHub repository.

➡️ **Download here**: [BCSDM_resources.zip](<https://www.dropbox.com/scl/fi/uubobpjwmvmkeq33rk858/BCSDM_resources.zip?rlkey=d6xehctpktpzzpebk01rt42hz&e=1&st=1z3559o6&dl=1>)

After downloading, unzip the file and place the contents in the root directory of the project:


## Run

### Train
```shell
python train.py --config ./configs/Euc_bc-deepovec.yml --device 0 --run Euc_bc-deepovec
python train.py --config ./configs/S2_bc-deepovec.yml --device 0 --run S2_bc-deepovec
python train.py --config ./configs/SE3_bc-deepovec.yml --device 0 --run SE3_bc-deepovec
```

### Evlauation(visualization)
```shell
python eval.py --config ./configs/eval/S2_bc-deepovec.yml --fit_traj --fit_all --mimic --contraction --cvf_mvf --vis_local --vis_sphere
python eval.py --config ./configs/eval/SE3_bc-deepovec.yml --fit_traj --fit_all --mimic --contraction --cvf_mvf
```

You can enable specific evaluation metrics and visualizations by passing the corresponding arguments:

| Argument         | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `--fit_traj`     | Evaluate trajectory fitting error                                           |
| `--fit_all`      | Evaluate fitting error in task space                                        |
| `--mimic`        | Evaluate mimicking error (difference from demonstration behavior)           |
| `--contraction`  | Compute contraction rate                                                    |
| `--cvf_mvf`      | Compute error between contacting vector field and mimicking vector field    |
| `--vis_local`    | Visualize the trained vector field in S2 local coordinates                  |
| `--vis_sphere`   | Visualize the trained vector field on the full S2 sphere                    |



### Simulation Interface

```bash
python sim.py
```

The `sim.py` script provides an interactive simulation environment to visualize how a robot behaves under the **BC-DeepOVEC** algorithm. This interface includes several interactive features:


| Control             | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `Start`             | Starts the robot's motion based on the BC-DeepOVEC policy                   |
| `Restart`           | Resets the simulation environment                                           |
| `Mouse disturbance` | Click and drag in the simulation window to apply manual external forces     |
| `disturbance scale` | Sets the magnitude of the random disturbance                                |
| `disturbance time`  | Sets the time at which the random disturbance is applied                    |
| `eta`               | Controls the contraction rate of the controller                             |
| `demo num`          | Selects which demonstration trajectory the robot should mimic               |
