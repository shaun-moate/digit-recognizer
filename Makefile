# TODO: create a local setup process to prompt for env variables and store them in .env

dvc/push:
	export POETRY_DOTENV_LOCATION=.env && poetry run -vvv python local-setup.py && poetry run -vvv dvc push

dvc/pull:
	export POETRY_DOTENV_LOCATION=.env && poetry run -vvv python local-setup.py && poetry run -vvv dvc pull
