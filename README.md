# Data Analysis in Earth Sciences

## Getting started
### Create virtual environment

Make sure you have `conda-lock` installed. If you already have it, run the command below to create the environment based on `conda-lock.yml` file.

```
conda-lock install --mamba -n daes-env conda-lock.yml
```

### Activate environment

```
mamba activate daes-env
```
### Requirements
1. Install the CDS API client
```
$ pip install "cdsapi>=0.7.4"
```
2. Register on the page: https://cds.climate.copernicus.eu/
3. Go to Your Profile and find your personal API Token
4. Open Notebook and paste:
```
url: https://cds.climate.copernicus.eu/api
key: YourAPIToken
```
5. Save file as: .cdsapirc
*For instructions on how to create a dot file on Windows, please see here or check the instructions here: https://gist.github.com/ozh/4131243


