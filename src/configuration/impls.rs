use {
    crate::{
        configuration::structs::{
            BuildConfig, ConfigConfig, Configuration, LoadConfig, OnnxLoadConfig,
        },
        section::add_section,
    },
    pyo3::{
        types::{PyAnyMethods, PyDict, PyList, PyListMethods},
        Bound, PyResult, Python,
    },
};

impl Configuration {
    pub fn explain(&self) {
        println!("\n[Configuration Summary]\n");
        // [config]
        add_section("config", &self.config);

        // [load]
        //
        match &self.load {
            LoadConfig::Onnx(load) => {
                add_section("load", load);
            }
        }

        // [build]
        add_section("build", &self.build);

        // [export]
        add_section("export", &self.export);
    }
}

impl ConfigConfig {
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
        dict.set_item("quantized_dtype", self.quantized_dtype.clone())?;
        dict.set_item("quantized_algorithm", self.quantized_algorithm.clone())?;
        dict.set_item("quantized_method", self.quantized_method.clone())?;
        dict.set_item("target_platform", self.target_platform.clone())?;
        set!(quant_img_RGB2BGR);
        set!(float_dtype);
        dict.set_item("optimization_level", self.optimization_level)?;
        set!(custom_string);
        set!(remove_weight);
        set!(compress_weight);
        set!(inputs_yuv_fmt);
        set!(single_core_mode);

        Ok(dict)
    }
}

impl OnnxLoadConfig {
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

impl BuildConfig {
    pub fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);

        macro_rules! set {
            ($field:ident) => {
                if let Some(ref val) = self.$field {
                    dict.set_item(stringify!($field), val)?;
                }
            };
        }

        set!(do_quantization);
        set!(dataset);
        set!(rknn_batch_size);
        set!(auto_hybrid);

        Ok(dict)
    }
}
