.PHONY: fmt
fmt:
	black .

.PHONY: lint
lint:
	black --check .
	flake8 .

.PHONY: test
test:
	pytest --verbose

.PHONY: test-quick
test-quick:
	pytest --verbose --benchmark-skip

.PHONY: install
install:
	pip install .

.PHONY: install-dev
install-dev:
	pip install -r requirements.txt \
	&& pip install -e .

.PHONY: clean-docs
clean-docs:
	rm -rf docs

.PHONY: docs
docs: clean-docs
	mkdir docs \
	&& pdoc --html --output-dir docs molesq
