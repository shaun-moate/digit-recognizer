# ml-bootstrap-example
Simple (and basic!) project set-up with an example implementation

# Getting Started 
Firstly, get yourself set up with dependencies by running `poetry shell`, getting you into a emulated shell environment with all dependencies installed.  To ensure you have everything installed you can run `poetry install`

Make sure to set up your `.env` file with the following:
- `AWS_ACCESS_KEY_ID`:  public key for aws
- `AWS_SECRET_ACCESS_KEY`:  private key for aws

If you need to get the data or want to ensure you have the latest data simply run `make dvc/pull`

Lastly, to run an experiment from e2e simply run `dvc exp run -f`

If you would like to change the parameters of the experiment, either edit `params.yaml` or run `dvc exp run -S <insert_params_you_want_to_change>`

Should be ready to go!
