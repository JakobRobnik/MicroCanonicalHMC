PKG_VERSION = $(shell python setup.py --version)

test:
	JAX_PLATFORM_NAME=cpu pytest --benchmark-disable
	mypy mclmc/sampling/sampler.py

set-bench:
	pytest --benchmark-autosave  

compare-bench:
	pytest --benchmark-compare=0001 --benchmark-compare-fail=mean:2%

# We launch the package release by tagging the master branch with the package's
# new version number.
release:
	git tag -a $(PKG_VERSION) -m $(PKG_VERSION)
	git push --tag

