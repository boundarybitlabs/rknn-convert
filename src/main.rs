use std::fs;

use pyo3::prelude::*;
use pyo3::BoundObject;
use pyo3::Python;

use crate::configuration::Configuration;
use crate::functions::call_rknn_build;
use crate::functions::call_rknn_config;
use crate::functions::call_rknn_export;
use crate::functions::call_rknn_load_onnx;

mod configuration;
mod functions;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string("config.toml")?;
    let config: Configuration = toml::from_str(&content)?;

    Python::with_gil(|py| {
        let rknn = py
            .import("rknn.api")?
            .getattr("RKNN")?
            .call0()?
            .into_bound();

        call_rknn_config(py, rknn.clone(), &config.config)?;

        match config.model {
            configuration::ModelLoading::Onnx(onnx_loading) => {
                call_rknn_load_onnx(py, rknn.clone(), &onnx_loading)?
            }
        }

        call_rknn_build(py, rknn.clone(), &config.build)?;
        call_rknn_export(py, rknn.clone(), &config.export)?;

        Ok(())
    })
}
