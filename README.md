# ml-bootstrap-example
Simple (and basic!) project set-up with an example implementation

# Getting Started 
Firstly, get yourself set up with dependencies by running `poetry shell`, getting you into a emulated shell environment with all dependencies installed.  To ensure you have everything installed you can run `poetry install`

Make sure to set up your `.env` file with the following:
- `AWS_ACCESS_KEY_ID`:  public key for aws
- `AWS_SECRET_ACCESS_KEY`:  private key for aws

Now! Get import environment variable set-up.  Have made this easy... all you need to run `make local/envs`.  Have a look at `Makefile` to see what it does.

Lastly, if you need to get the data or want to ensure you have the latest data simply run `dvc pull`

Should be ready to go!
