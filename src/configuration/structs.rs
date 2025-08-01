use {
    schemars::JsonSchema,
    serde::{Deserialize, Serialize},
    validator::Validate,
};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Configuration {
    pub config: ConfigConfig,
    pub load: LoadConfig,
    pub build: BuildConfig,
    pub export: ExportConfig,
}

impl Default for ConfigConfig {
    fn default() -> Self {
        ConfigConfig {
            mean_values: None,
            std_values: None,
            quantized_dtype: "asymmetric_quantized-8".to_string(),
            quantized_algorithm: "normal".to_string(),
            quantized_method: "channel".to_string(),
            target_platform: None,
            quant_img_RGB2BGR: Some(false),
            float_dtype: Some("float16".to_string()),
            optimization_level: 3,
            custom_string: None,
            remove_weight: Some(false),
            compress_weight: Some(false),
            inputs_yuv_fmt: None,
            single_core_mode: Some(false),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Validate, JsonSchema)]
#[serde(default)]
pub struct ConfigConfig {
    /// Mean values for normalization.
    pub mean_values: Option<Vec<f32>>,

    /// Standard deviation values for normalization.
    pub std_values: Option<Vec<f32>>,

    /// Quantized dtype. [asymmetric_quantized-8]
    #[validate(custom(function = "validate_quantized_dtype"))]
    pub quantized_dtype: String,

    /// Quantized algorithm. [normal, mmse]
    #[validate(custom(function = "validate_quantized_algorithm"))]
    pub quantized_algorithm: String,

    /// Quantized method. [channel, group, layers]
    #[validate(custom(function = "validate_quantized_method"))]
    pub quantized_method: String,

    /// target platform [rk3588, rk3576, rk3568, rk3562, rk2118, rv1126B, rv1106, ...]
    #[validate(length(min = 1))]
    pub target_platform: Option<String>,

    /// BGR channel order instead of RGB
    pub quant_img_RGB2BGR: Option<bool>,

    /// float_dtype [float16, float32, bfloat16]
    #[validate(custom(function = "validate_float_dtype"))]
    pub float_dtype: Option<String>,

    /// Optimization level [0, 1, 2, 3]
    #[validate(range(min = 0, max = 3))]
    pub optimization_level: u8,

    /// A custom string to add to the model.
    pub custom_string: Option<String>,
    /// Remove the weights from the model.
    pub remove_weight: Option<bool>,
    /// Compress the model weights.
    pub compress_weight: Option<bool>,
    /// Input YUV format.
    pub inputs_yuv_fmt: Option<String>,
    /// Single core mode.
    pub single_core_mode: Option<bool>,
}

fn validate_quantized_dtype(value: &str) -> Result<(), validator::ValidationError> {
    match value {
        "asymmetric_quantized-8" => Ok(()),
        _ => Err(validator::ValidationError::new("invalid_quantized_dtype")),
    }
}

fn validate_quantized_algorithm(value: &str) -> Result<(), validator::ValidationError> {
    match value {
        "normal" | "mmse" => Ok(()),
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, JsonSchema)]
#[serde(tag = "model_type")]
pub enum LoadConfig {
    Onnx(OnnxLoadConfig),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(default)]
pub struct OnnxLoadConfig {
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

impl Default for OnnxLoadConfig {
    fn default() -> Self {
        Self {
            model: "model.onnx".to_string(),
            inputs: None,
            input_size_list: None,
            input_initial_val_file: None,
            outputs: None,
        }
    }
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self::Onnx(OnnxLoadConfig::default())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, JsonSchema)]
#[serde(default)]
pub struct BuildConfig {
    /// Whether to quantize the model (default: true)
    pub do_quantization: Option<bool>,

    /// Path to dataset file for quantization (default: None)
    pub dataset: Option<String>,

    /// Batch size for inference (default: None)
    pub rknn_batch_size: Option<i32>,

    /// Enable automatic hybrid quantization (default: false)
    pub auto_hybrid: Option<bool>,
}

impl Default for BuildConfig {
    fn default() -> Self {
        Self {
            do_quantization: Some(true),
            dataset: None,
            rknn_batch_size: None,
            auto_hybrid: Some(false),
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Deserialize, JsonSchema)]
#[serde(default)]
pub struct ExportConfig {
    pub export_path: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self { export_path: None }
    }
}
