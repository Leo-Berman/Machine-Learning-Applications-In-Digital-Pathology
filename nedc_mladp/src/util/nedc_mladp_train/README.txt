Command used to launch a job to GPU:
sbatch -p gpu --gres=gpu -w nedc_012 run_train.sh

Additionally, I think you can also ssh into nedc_012 and run directly from there. 
Running the code on CPU might take more than 20 minutes.