.PHONY: run
run:
	pixi run python -m pyvenom.dependency_tree

.PHONY: test
test:
	pixi run -e test pytest -sx tests/


.PHONY: debug
debug:
	pixi run -e test pytest -vx  -m debug  tests/

.PHONY: cov
cov:
	pixi run -e test pytest tests/ --cov=ididi --cov-report term-missing 


.PHONY: visualize
visualize:
	pixi run -e dev python -m illustrate


.PHONY: patch
patch:
	pixi run -e publish hatch version patch

.PHONY: build
build:
	pixi run -e publish hatch build

.PHONY: publish
publish:
	pixi run -e publish publish

# .PHONY: release
# release:
# 	git tag -a v0.1.2 -m "Release version 0.1.2"
# 	git push origin master
# 	git push origin v0.1.2
