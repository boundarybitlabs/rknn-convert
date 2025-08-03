use {
    schemars::{schema_for, JsonSchema},
    serde::Serialize,
    serde_json::Value,
    stanza::{
        renderer::{console::Console, Renderer},
        table::Table,
    },
};

/// Helper to get a field's value from an instance by its string name.
fn get_field_value<T: Serialize>(instance: &T, key: &str) -> Value {
    let json_value = serde_json::to_value(instance).unwrap_or(Value::Null);
    json_value
        .as_object()
        .and_then(|obj| obj.get(key).cloned())
        .unwrap_or(Value::Null)
}

/// Helper to format a JSON Value for cleaner table output.
fn format_value(value: &Value) -> String {
    if let Some(s) = value.as_str() {
        s.to_string()
    } else {
        value.to_string()
    }
}

pub fn add_section<T>(title: &str, instance: &T)
where
    T: JsonSchema + Serialize + Default + PartialEq,
{
    let default_instance = T::default();
    // Convert the schema to a serde_json::Value to work with it directly.
    let schema_value = serde_json::to_value(schema_for!(T)).unwrap();

    let mut table = Table::default();
    table.push_row(["Key", "Value", "Default", "Set by User?", "Description"]);

    // Get the "properties" map from the top-level schema value.
    if let Some(properties_map) = schema_value.get("properties").and_then(Value::as_object) {
        // Iterate directly over the properties map (key, value pairs).
        for (key, field_schema_value) in properties_map {
            let value = get_field_value(instance, key);
            let default_value = get_field_value(&default_instance, key);
            let is_set = value != default_value;

            // Get the description by looking up the "description" key
            // in the field's schema value.
            let description = field_schema_value
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or("") // Use an empty string if not present
                .to_string();

            table.push_row([
                key.as_str(),
                &format_value(&value),
                &format_value(&default_value),
                if is_set { "Yes" } else { "No" },
                &description,
            ]);
        }
    }

    let console = Console::default();
    println!("[{}]", title);
    println!("{}", console.render(&table));
    println!();
}
