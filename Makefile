env:
		python3 -m venv ~/my-env
		source ~/my-env/bin/activate
		file $(which python)
		python --version
dev:
		python -m pip install -U pip setuptools wheel pip-tools isort black flake8 pylint mypy

install:
		python -m pip install -e .
requirements:
		python -m piptools compile --no-emit-index-url
# https://code.visualstudio.com/docs/python/linting
lint:
		python -m flake8 .
		python -m pylint .
		python -m mypy .
test: lint
		cd tests
		python -m pip install -r requirements.txt
		python -m pytest
format:
		python -m isort .
		python -m black .
