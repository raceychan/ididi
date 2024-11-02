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

.PHONY: minor
minor:
	pixi run -e publish hatch version minor

.PHONY: release

VERSION ?= x.x.x
BRANCH = version/$(VERSION)

release: check-branch check-version update-version hatch-build git-merge git-tag git-push

check-branch:
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "$(BRANCH)" ]; then \
		echo "Error: You must be on branch $(BRANCH) to release."; \
		exit 1; \
	fi

check-version:
	@CURRENT_VERSION=$$(pixi run -e publish hatch version); \
	if [ "$$CURRENT_VERSION" = "" ]; then \
		echo "Error: Unable to retrieve current version."; \
		exit 1; \
	fi; \
	CURRENT_MAJOR=$${CURRENT_VERSION%%.*}; \
	CURRENT_MINOR=$${CURRENT_VERSION#*.}; \
	CURRENT_MINOR=$${CURRENT_MINOR%%.*}; \
	CURRENT_PATCH=$${CURRENT_VERSION##*.}; \
	NEW_MAJOR=$${VERSION%%.*}; \
	NEW_MINOR=$${VERSION#*.}; \
	NEW_MINOR=$${NEW_MINOR%%.*}; \
	NEW_PATCH=$${VERSION##*.}; \
	if [ "$$NEW_MAJOR" -gt "$$CURRENT_MAJOR" ] || ( \
		[ "$$NEW_MAJOR" -eq "$$CURRENT_MAJOR" ] && [ "$$NEW_MINOR" -gt "$$CURRENT_MINOR" ] ) || ( \
		[ "$$NEW_MAJOR" -eq "$$CURRENT_MAJOR" ] && [ "$$NEW_MINOR" -eq "$$CURRENT_MINOR" ] && [ "$$NEW_PATCH" -gt "$$CURRENT_PATCH" ] ); then \
		echo "Version $(VERSION) is valid."; \
	else \
		echo "Error: The new version $(VERSION) must be larger than the current version $$CURRENT_VERSION."; \
		exit 1; \
	fi

update-version:
	@echo "Updating Pixi version to $(VERSION)..."
	@pixi run -e publish hatch version $(VERSION)

hatch-build:
	@echo "Building version $(VERSION)..."
	@pixi run -e publish hatch build

git-merge:
	@echo "Merging $(BRANCH) into master..."
	@git checkout master
	@git merge "$(BRANCH)"

git-tag:
	@echo "Tagging the release..."
	@git tag -a "v$(VERSION)" -m "Release version $(VERSION)"

git-push:
	@echo "Pushing to remote repository..."
	@git push origin master
	@git push origin "v$(VERSION)"