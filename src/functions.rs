use {
    crate::configuration::structs::{
        AccuracyAnalysisConfig, BuildConfig, ConfigConfig, ExportConfig, OnnxLoadConfig,
    },
    pyo3::{types::PyAnyMethods, Bound, PyAny, PyResult, Python},
};

pub fn call_rknn_config(
    py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    config: &ConfigConfig,
) -> PyResult<()> {
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
    onnx: &OnnxLoadConfig,
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

pub fn call_rknn_build(
    py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    build: &BuildConfig,
) -> PyResult<()> {
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

pub fn call_rknn_accuracy_analysis(
    py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    accuracy_analysis: Option<&AccuracyAnalysisConfig>,
) -> PyResult<()> {
    let Some(accuracy_analysis) = accuracy_analysis else {
        return Ok(());
    };

    if !accuracy_analysis.should_run() {
        return Ok(());
    }

    let kwargs = accuracy_analysis.to_pydict(py)?;
    let result = rknn.call_method("accuracy_analysis", (), Some(&kwargs))?;
    let code: i32 = result.extract()?;

    if code != 0 {
        Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "rknn.accuracy_analysis failed with code {}",
            code
        )))
    } else {
        Ok(())
    }
}

pub fn call_rknn_export(
    _py: Python<'_>,
    rknn: Bound<'_, PyAny>,
    export: &ExportConfig,
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
