module load cudatoolkit/11.7
module load cudnn/8.9.1_cuda11
module load python

conda create -n mchmc python=3.9 pip numpy scipy matplotlib pandas

conda activate mchmc

pip install --no-cache-dir "jax[cuda11_cudnn82]==0.4.7" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html