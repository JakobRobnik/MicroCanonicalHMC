test:
	JAX_PLATFORM_NAME=cpu pytest --benchmark-disable
	mypy mclmc/sampler.py

set-bench:
	pytest --benchmark-autosave  

compare-bench:
	pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:2%
