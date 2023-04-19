# Install package packages
install-reqs:
	pip install -r requirements.txt

# Install test packages
install-test-reqs:
	pip install -r test-requirements.txt

# Linting script
lint:
	scripts/lint.sh


env:
	sh ./scripts/create_env.sh


