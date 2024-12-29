.PHONY: test
test:
	pixi run -e test pytest -m "not benchmark" -vsx tests/

.PHONY: debug
debug:
	pixi run -e test pytest -vx  -m debug  tests/

.PHONY: feat
feat:
	pixi run -e test pytest -vx  tests/test_feat.py

.PHONY: benchmark
benchmark:
	pixi run -e test pytest -m benchmark tests/

.PHONY: debug-cov
debug-cov:
	pixi run -e test pytest -vx  -m debug --cov=ididi --cov-report term-missing tests/

.PHONY: cov
cov:
	pixi run -e test pytest -m "not benchmark" tests/ --cov=ididi --cov-report term-missing 

.PHONY: report
report:
	pixi run -e test pytest tests/ --cov=ididi --cov-report html


.PHONY: docs
docs:
	pixi run -e dev mkdocs serve

# =============

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

release: check-branch check-version update-version git-commit git-merge git-tag git-push hatch-build 

check-branch:
	@if [ "$$(git rev-parse --abbrev-ref HEAD)" != "$(BRANCH)" ]; then \
		echo "Error: You must be on branch $(BRANCH) to release."; \
		echo "Did you forget to provide VERSION?"; \
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



git-commit:
	@echo "Committing changes..."
	@git add -A
	@git commit -m "Release version $(VERSION)"

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

hatch-build:
	@echo "Building version $(VERSION)..."
	@pixi run -e publish hatch build

pypi-release:
	pixi run -e publish publish

delete-branch:
	git branch -d $(BRANCH)
	git push origin --delete $(BRANCH)

