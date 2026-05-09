# rknn-convert
Command-line tool for converting machine learning model files to RKNN format.

## Purpose
Using the rknn-toolkit2 library requires writing the same function calls in a similar order every time.
This tool simplifies the process by allowing you to define your conversion process in a TOML configuration file.

## Usage
```
rknn-convert convert <config.toml>
```
Converts the model to RKNN format using the configuration file.

```
rknn-convert explain <config.toml>
```
Shows each configuration option and whether is is the default value.

Pipx installable via `pipx install rknn-convert`
