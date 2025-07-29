#![allow(non_snake_case)]
#![allow(dead_code)]

use pyo3::prelude::*;
use pyo3::types::{PyDictMethods, PyList};
use pyo3::{types::PyDict, PyResult, Python};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Configuration {
    pub config: Config,
    pub model: ModelLoading,
}

use std::collections::HashMap;
use validator::Validate;

#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct Config {
    pub mean_values: Option<Vec<f32>>,

    pub std_values: Option<Vec<f32>>,

    #[validate(custom(function = "validate_quantized_dtype"))]
    pub quantized_dtype: Option<String>,

    #[validate(custom(function = "validate_quantized_algorithm"))]
    pub quantized_algorithm: Option<String>,

    #[validate(custom(function = "validate_quantized_method"))]
    pub quantized_method: Option<String>,

    #[validate(length(min = 1))]
    pub target_platform: Option<String>,

    pub quant_img_RGB2BGR: Option<bool>,

    #[validate(custom(function = "validate_float_dtype"))]
    pub float_dtype: Option<String>,

    #[validate(range(min = 0, max = 3))]
    pub optimization_level: Option<u8>,

    pub custom_string: Option<String>,
    pub remove_weight: Option<bool>,
    pub compress_weight: Option<bool>,
    pub inputs_yuv_fmt: Option<String>,
    pub single_core_mode: Option<bool>,
    pub model_pruning: Option<bool>,
    pub op_target: Option<HashMap<String, String>>,
    pub dynamic_input: Option<Vec<Vec<Vec<usize>>>>,
    pub quantize_weight: Option<bool>,
    pub remove_reshape: Option<bool>,
    pub sparse_infer: Option<bool>,
    pub enable_flash_attention: Option<bool>,
    pub auto_hybrid_cos_thresh: Option<f32>,
    pub auto_hybrid_euc_thresh: Option<f32>,
}

fn validate_quantized_dtype(value: &str) -> Result<(), validator::ValidationError> {
    match value {
        "w8a8" | "w8a16" | "w16a16i" | "w16a16i_dfp" | "w4a16" => Ok(()),
        _ => Err(validator::ValidationError::new("invalid_quantized_dtype")),
    }
}

fn validate_quantized_algorithm(value: &str) -> Result<(), validator::ValidationError> {
    match value {
        "normal" | "mmse" | "kl_divergence" | "gdq" => Ok(()),
        _ => Err(validator::ValidationError::new(
            "invalid_quantized_algorithm",
        )),
    }
}

fn validate_quantized_method(value: &str) -> Result<(), validator::ValidationError> {
    if value == "layer" || value == "channel" || value.starts_with("group") {
        Ok(())
    } else {
        Err(validator::ValidationError::new("invalid_quantized_method"))
    }
}

fn validate_float_dtype(value: &str) -> Result<(), validator::ValidationError> {
    match value {
        "float16" => Ok(()),
        _ => Err(validator::ValidationError::new("invalid_float_dtype")),
    }
}

impl Config {
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        macro_rules! set {
            ($key:ident) => {
                if let Some(ref v) = self.$key {
                    dict.set_item(stringify!($key), v)?;
                }
            };
            ($key:ident, $conv:expr) => {
                if let Some(ref v) = self.$key {
                    dict.set_item(stringify!($key), $conv(v)?)?;
                }
            };
        }

        // Vec<f32>
        set!(mean_values, |v: &Vec<f32>| PyList::new(py, v));
        set!(std_values, |v: &Vec<f32>| PyList::new(py, v));

        // Strings and basic types
        set!(quantized_dtype);
        set!(quantized_algorithm);
        set!(quantized_method);
        set!(target_platform);
        set!(quant_img_RGB2BGR);
        set!(float_dtype);
        set!(optimization_level);
        set!(custom_string);
        set!(remove_weight);
        set!(compress_weight);
        set!(inputs_yuv_fmt);
        set!(single_core_mode);
        set!(model_pruning);
        set!(quantize_weight);
        set!(remove_reshape);
        set!(sparse_infer);
        set!(enable_flash_attention);
        set!(auto_hybrid_cos_thresh);
        set!(auto_hybrid_euc_thresh);

        // op_target: HashMap<String, String>
        if let Some(ref map) = self.op_target {
            let py_map = PyDict::new(py);
            for (k, v) in map {
                py_map.set_item(k, v)?;
            }
            dict.set_item("op_target", py_map)?;
        }

        // dynamic_input: Vec<Vec<Vec<usize>>>
        if let Some(ref outer) = self.dynamic_input {
            let py_outer = PyList::empty(py);
            for shape_set in outer {
                let py_set = PyList::empty(py);
                for shape in shape_set {
                    py_set.append(PyList::new(py, shape)?)?;
                }
                py_outer.append(py_set)?;
            }
            dict.set_item("dynamic_input", py_outer)?;
        }

        Ok(dict)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ModelLoading {
    Onnx(OnnxLoading),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnnxLoading {
    /// Path to the ONNX model file
    pub model: String,

    /// Optional list of input names (e.g. ["input0"])
    pub inputs: Option<Vec<String>>,

    /// Optional shape list matching each input (e.g. [[1, 3, 224, 224]])
    pub input_size_list: Option<Vec<Vec<usize>>>,

    /// Path to a `.npy` or `.npz` file containing a list of NumPy arrays
    pub input_initial_val_file: Option<String>,

    /// Optional list of output names
    pub outputs: Option<Vec<String>>,
}

impl OnnxLoading {
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        macro_rules! set {
            ($key:ident) => {
                if let Some(ref v) = self.$key {
                    dict.set_item(stringify!($key), v)?;
                }
            };
            ($key:ident, $conv:expr) => {
                if let Some(ref v) = self.$key {
                    dict.set_item(stringify!($key), $conv(v)?)?;
                }
            };
        }

        // inputs and outputs: Vec<String> → PyList
        set!(inputs, |v: &Vec<String>| PyList::new(py, v));
        set!(outputs, |v: &Vec<String>| PyList::new(py, v));

        // input_size_list: Vec<Vec<usize>> → PyList<PyList>
        if let Some(ref shape_list) = self.input_size_list {
            let outer = PyList::empty(py);
            for shape in shape_list {
                outer.append(PyList::new(py, shape)?)?;
            }
            dict.set_item("input_size_list", outer)?;
        }

        // input_initial_val_file: load .npy or .npz → PyList of arrays
        if let Some(ref path) = self.input_initial_val_file {
            let np = py.import("numpy")?;
            let obj = np.getattr("load")?.call1((path,))?;

            if obj.hasattr("files")? {
                // .npz: extract arrays by key
                let mut arrays = Vec::new();
                for key in obj.getattr("files")?.try_iter()? {
                    arrays.push(obj.get_item(key?)?);
                }
                dict.set_item("input_initial_val", PyList::new(py, arrays)?)?;
            } else {
                // .npy: single array
                dict.set_item("input_initial_val", PyList::new(py, &[obj])?)?;
            }
        }

        Ok(dict)
    }
}
