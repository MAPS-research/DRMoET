#!/bin/bash

#SBATCH --job-name=YourNameHere
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=6:00:00         # max time for running this job
#SBATCH --gres=gpu:1           # request for any gpu available
##SBATCH --gres=gpu:rtx8000:4     # [alternative]request for gpu(s) with specific type
##SBATCH --output=log/slurm-%j.out # [optional] output log file, need to create a folder named log

# Singularity path

# Load site configuration (run from the repository root)
source scripts/config.sh

ext3_path="${SINGULARITY_OVERLAY:?set SINGULARITY_OVERLAY in scripts/config.sh}"
sif_path="${SINGULARITY_IMAGE:?set SINGULARITY_IMAGE in scripts/config.sh}"

# start running
singularity exec --nv \
--overlay ${ext3_path}:ro \
${sif_path} /bin/bash -c "
    source /ext3/env.sh
    REPLACE THIS WITH YOUR COMMANDS
"