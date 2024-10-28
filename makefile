.PHONY: run
run:
	pixi run python -m pyvenom.dependency_tree

.PHONY: test
test:
	pixi run -e test pytest -sx tests/


.PHONY: cov
cov:
	pixi run -e test pytest tests/ --cov=ididi --cov-report term-missing 


.PHONY: visualize
visualize:
	pixi run -e dev python -m illustrate
