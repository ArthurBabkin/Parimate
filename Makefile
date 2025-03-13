init:
	python -m pip install poetry
	poetry install --no-root

test:
	poetry run pytest -sv tests

run:
	poetry run python cmd/app/main.py --config_path config/config.yaml
