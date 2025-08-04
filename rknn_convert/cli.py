import typer
from pathlib import Path
import rknn_convert_inner

app = typer.Typer(help="A tool for converting ONNX,TF,PyTorch models to RKNN models")

def validate_config_file(ctx: typer.Context, param: typer.CallbackParam, value: Path):
    if not value.exists():
        raise typer.BadParameter(f"Config file {value} does not exist.")
    return value

@app.command()
def convert(
    config: Path = typer.Argument(..., help="Path to the TOML configuration file", callback=validate_config_file)
):
    """
    Convert the model to RKNN format based on the TOML configuration file.
    """

    try:
        rknn_convert_inner.rust_convert(str(config))
    except Exception as e:
        typer.echo(f"Error: {e}")

@app.command()
def explain(
    config: Path = typer.Argument(..., help="Path to the TOML configuration file", callback=validate_config_file)
):
    """
    Explain the model based on the TOML configuration file.
    """

    try:
        rknn_convert_inner.rust_explain(str(config))
    except Exception as e:
        typer.echo(f"Error: {e}")

def main():
    app()
