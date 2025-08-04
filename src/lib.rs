use {
    crate::{
        configuration::structs::Configuration,
        functions::{call_rknn_build, call_rknn_config, call_rknn_export, call_rknn_load_onnx},
    },
    pyo3::{exceptions::PyRuntimeError, prelude::*, BoundObject, Python},
    std::fs,
};

mod configuration;
mod functions;
mod section;

#[pymodule]
fn rknn_convert_inner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_convert, m)?)?;
    m.add_function(wrap_pyfunction!(rust_explain, m)?)?;
    Ok(())
}

#[pyfunction]
fn rust_convert(path: String) -> PyResult<()> {
    rust_convert_inner(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

#[pyfunction]
fn rust_explain(path: String) -> PyResult<()> {
    rust_explain_inner(path).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    Ok(())
}

fn rust_convert_inner(path: String) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(&path)?;
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
        Ok::<_, Box<dyn std::error::Error>>(())
    })?;

    Ok(())
}

fn rust_explain_inner(path: String) -> Result<(), Box<dyn std::error::Error>> {
    let content = fs::read_to_string(path)?;
    let config: Configuration = toml::from_str(&content)?;
    config.explain();
    Ok(())
}
