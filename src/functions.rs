use pyo3::types::PyAnyMethods;
use pyo3::{Bound, PyAny, PyResult, Python};

use crate::configuration::{Build, Config, ExportRknn, OnnxLoading};

pub fn call_rknn_config(py: Python<'_>, rknn: Bound<'_, PyAny>, config: &Config) -> PyResult<()> {
    let config_dict = config.to_pydict(py)?;
    let result = rknn.call_method("config", (), Some(&config_dict))?;
    let code: i32 = result.extract()?;

    if code != 0 {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rknn.config failed with code {}",
            code
        )))
    } else {
        Ok(())
    }
}

pub fn call_rknn_load_onnx(
    py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    onnx: &OnnxLoading,
) -> PyResult<()> {
    let kwargs = onnx.to_pydict(py)?;
    let result = rknn.call_method("load_onnx", (&onnx.model,), Some(&kwargs))?;
    let code: i32 = result.extract()?;

    if code != 0 {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rknn.load_onnx failed with code {}",
            code
        )))
    } else {
        Ok(())
    }
}

pub fn call_rknn_build(py: Python<'_>, rknn: Bound<'_, PyAny>, build: &Build) -> PyResult<()> {
    let kwargs = build.to_pydict(py)?;
    let result = rknn.call_method("build", (), Some(&kwargs))?;
    let code: i32 = result.extract()?;

    if code != 0 {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rknn.build failed with code {}",
            code
        )))
    } else {
        Ok(())
    }
}

pub fn call_rknn_export(
    py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    export: &ExportRknn,
) -> PyResult<()> {
    let result = rknn.call_method("export_rknn", (&export.export_path,), None)?;
    let code: i32 = result.extract()?;

    if code != 0 {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rknn.export_rknn failed with code {}",
            code
        )))
    } else {
        Ok(())
    }
}
