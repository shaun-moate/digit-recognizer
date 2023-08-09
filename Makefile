# TODO: create a local setup process to prompt for env variables and store them in .env

local/envs:
	export POETRY_DOTENV_LOCATION=.env && poetry run python local-setup.py
