use std::io;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let request: serde_json::Value = serde_json::from_reader(io::stdin().lock())?;
    let value = request
        .get("value")
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "value must be an integer"))?;
    let next = value
        .checked_add(1)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "value overflows i64"))?;
    serde_json::to_writer(io::stdout().lock(), &serde_json::json!({"value": next}))?;
    Ok(())
}
