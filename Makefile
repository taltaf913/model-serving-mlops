# Install pip packages
install-reqs:
	pip install -r requirements.txt

install-test-reqs:
	pip install -r test-requirements.txt

# Lint
lint:
	scripts/lint.sh