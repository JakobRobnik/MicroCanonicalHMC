from benchmarks.inference_models import *


model = Gaussian(ndims=100, eigenvalues='Gamma', numpy_seed= rng_inference_gym_icg)

eigs = model.E_x2

print(np.max(eigs)/np.min(eigs))
print(np.max(np.square(model.E_x2)/model.Var_x2))
