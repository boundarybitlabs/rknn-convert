use {
    crate::{
        configuration::structs::Configuration,
        functions::{call_rknn_build, call_rknn_config, call_rknn_export, call_rknn_load_onnx},
    },
    clap::{Parser, Subcommand},
    pyo3::{prelude::*, BoundObject, Python},
    std::fs,
};

mod configuration;
mod functions;
mod section;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.cmd {
        Action::Convert(convert) => {
            let content = fs::read_to_string(convert.config)?;
            let config: Configuration = toml::from_str(&content)?;

            Python::with_gil(|py| {
                let rknn = py
                    .import("rknn.api")?
                    .getattr("RKNN")?
                    .call0()?
                    .into_bound();

                call_rknn_config(py, rknn.clone(), &config.config)?;

                match config.load {
                    configuration::structs::LoadConfig::Onnx(onnx_loading) => {
                        call_rknn_load_onnx(py, rknn.clone(), &onnx_loading)?
                    }
                }

                call_rknn_build(py, rknn.clone(), &config.build)?;
                call_rknn_export(py, rknn.clone(), &config.export)?;

                Ok(())
            })
        }
        Action::Explain(explain) => {
            let content = fs::read_to_string(explain.config)?;
            let config: Configuration = toml::from_str(&content)?;
            config.explain();

            Ok(())
        }
    }
}

#[derive(Debug, Parser)]
pub struct Args {
    #[clap(help = "The command to perform.")]
    #[clap(subcommand)]
    pub cmd: Action,
}

#[derive(Debug, Subcommand, Clone)]
pub enum Action {
    Explain(Explanation),
    Convert(Convert),
}

#[derive(Debug, Parser, Clone)]
pub struct Convert {
    #[clap(help = "Path to the TOML configuration file.")]
    pub config: String,
}

#[derive(Debug, Parser, Clone)]
pub struct Explanation {
    #[clap(help = "Path to the TOML configuration file.")]
    pub config: String,
}
