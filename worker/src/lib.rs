use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use std::f64;
use std::collections::HashMap;
use half::f16;
use web_sys::{js_sys, Request, RequestInit, RequestMode, Response, console};
use std::cell::RefCell;
use js_sys::{ArrayBuffer, Uint8Array};
use console_error_panic_hook;
use std::panic;


thread_local! {
    static GLOBAL_MAP: RefCell<HashMap<String, Vec<f16>>> = RefCell::new(HashMap::new());
}

// Calculate the cosine similarity between two vectors
fn cosine_similarity(repr1: &[f64], repr2: &[f64]) -> f64 {
    let dot_product: f64 = repr1.iter().zip(repr2.iter()).map(|(a, b)| a * b).sum();
    let norm_a: f64 = repr1.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
    let norm_b: f64 = repr2.iter().map(|b| b.powi(2)).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        panic!("One of the vectors is zero");
    }

    dot_product / (norm_a * norm_b)
}

// Calculate the centered cosine similarity between two vectors
fn centered_cosine_similarity(repr1: &[f64], means1: &[f64], repr2: &[f64], means2: &[f64]) -> f64 {
    let mean: Vec<f64> = means1.iter().zip(means2.iter()).map(|(a, b)| (a + b) / 2.0).collect();
    let adjusted_repr1: Vec<f64> = repr1.iter().zip(mean.iter()).map(|(r, m)| r - m).collect();
    let adjusted_repr2: Vec<f64> = repr2.iter().zip(mean.iter()).map(|(r, m)| r - m).collect();

    cosine_similarity(&adjusted_repr1, &adjusted_repr2)
}

// Calculate the Euclidean distance between two vectors
fn euclidean_distance(repr1: &[f64], repr2: &[f64]) -> f64 {
    repr1.iter().zip(repr2.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
}

// Calculate the Manhattan distance between two vectors
fn manhattan_distance(repr1: &[f64], repr2: &[f64]) -> f64 {
    repr1.iter().zip(repr2.iter()).map(|(a, b)| (a - b).abs()).sum()
}

// Calculate the Chebyshev distance between two vectors
fn chebyshev_distance(repr1: &[f64], repr2: &[f64]) -> f64 {
    repr1.iter().zip(repr2.iter()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
}


fn get_repr(url: String) -> Option<Vec<f64>> {
    GLOBAL_MAP.with(|map| {
        map.borrow().get(&url).map(|vec| vec.iter().map(|&num| num.to_f64()).collect())
    })
}



#[wasm_bindgen]
pub fn calc_similarities(func: String, repr1_str: String, repr2_str: String, step1: usize, step2: usize, row: usize, col: usize, n: usize, m: usize) -> Result<Vec<f64>, JsValue> {
    console_error_panic_hook::set_once();  // better error messages in the console

    let repr1: Vec<f64> = match get_repr(repr1_str.clone()) {
        Some(repr) => repr,
        None => return Err(JsValue::from_str(&format!("Failed to get representation 1, url: {}", repr1_str))),
    };
    let repr2: Vec<f64> = match get_repr(repr2_str.clone()) {
        Some(repr) => repr,
        None => return Err(JsValue::from_str(&format!("Failed to get representation 2, url: {}", repr2_str))),
    };

    // parse func
    let similarity_fn = match func.as_str() {
        "cosine" => cosine_similarity,
        // "centered_cosine" => centered_cosine_similarity,
        "euclidean" => euclidean_distance,
        "manhattan" => manhattan_distance,
        "chebyshev" => chebyshev_distance,
        _ => panic!("Unknown similarity function"),
    };

    // Calculate similarities
    let mut similarities = Vec::new();
    let base_offset = (step1 * n * n + row * n + col) * m;
    for j in 0..n {
        for i in 0..n {
            let base_slice = &repr1[base_offset..base_offset + m];
            let concept_offset = (step2 * n * n + i * n + j) * m;
            let concept_slice = &repr2[concept_offset..concept_offset + m];

            // Calculate similarity using the provided function
            let similarity = similarity_fn(base_slice, concept_slice);
            similarities.push(similarity);
        }
    }

    // convert distance to similarity for euclidean, manhattan, and chebyshev
    if func == "euclidean" || func == "manhattan" || func == "chebyshev" {
        let max_distance = similarities.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        similarities = similarities.iter().map(|&d| 1.0 - d / max_distance).collect();
    }

    Ok(similarities)
}


#[wasm_bindgen]
pub async fn fetch_repr(url: String) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();  // better error messages in the console

    // if the representation is already fetched, return
    if GLOBAL_MAP.with(|map| map.borrow().contains_key(&url)) {
        return Ok(());
    }

    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(&url, &opts)?;
    let global = js_sys::global().unchecked_into::<web_sys::WorkerGlobalScope>();
    let resp_value = JsFuture::from(global.fetch_with_request(&request)).await?;
    let resp: Response = match resp_value.dyn_into() {
        Ok(resp) => resp,
        Err(_) => return Err(JsValue::from_str("Failed to get Response: resp_value.dyn_into() failed")),
    };

    let buffer_value = JsFuture::from(resp.array_buffer()?).await?;
    let buffer: ArrayBuffer = match buffer_value.dyn_into() {
        Ok(buffer) => buffer,
        Err(_) => return Err(JsValue::from_str("Failed to get ArrayBuffer: buffer_value.dyn_into() failed")),
    };
    if buffer.byte_length() % 4 != 0 {
        return Err(JsValue::from_str("Buffer length is not a multiple of 4"));
    }
    let bytes = Uint8Array::new(&buffer).to_vec();
    let float16_data: Vec<f16> = bytes.chunks(4).map(|chunk| f16::from_f32(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))).collect();

    GLOBAL_MAP.with(|map| {
        map.borrow_mut().insert(url, float16_data);
    });

    // log currently available representations
    // let reprs: Vec<String> = GLOBAL_MAP.with(|map| map.borrow().keys().cloned().collect());
    // console::log_1(&JsValue::from_str(&format!("Currently available representations: {:?}", reprs)));

    Ok(())
}
