use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use std::collections::HashMap;
use half::f16;
use web_sys::{js_sys, Request, RequestInit, RequestMode, Response, console};
use std::cell::RefCell;
use js_sys::{ArrayBuffer, Uint8Array};
use console_error_panic_hook;
use std::panic;
use ndarray::{s, ArrayView1, ArrayView4};
use std::sync::Arc;


// cache to store representations
thread_local! {
    static GLOBAL_MAP: RefCell<HashMap<String, Arc<Vec<f32>>>> = RefCell::new(HashMap::new());
}

// insert representation into cache
fn insert_repr(url: &str, data: Vec<f32>) {
    GLOBAL_MAP.with(|map| {
        map.borrow_mut().insert(url.to_string(), Arc::new(data));
    });
}

// get representation from cache
fn get_repr(url: &str, shape: (usize, usize, usize, usize)) -> Option<ArrayView4<'static, f32>> {
    GLOBAL_MAP.with(|map| {
        map.borrow().get(url).map(|arc_vec| {
            let static_slice: &'static [f32] = unsafe {
                std::mem::transmute(&arc_vec[..])
            };
            ArrayView4::from_shape(shape, static_slice).unwrap()
        })
    })
}

#[wasm_bindgen]
pub fn calc_similarities(
    func: String,
    repr1_str: String,
    repr2_str: String,
    step1: usize,
    step2: usize,
    row: usize,
    col: usize,
    steps: usize,
    n: usize,
    m: usize,
) -> Result<Vec<f32>, JsValue> {
    console_error_panic_hook::set_once();
    let time_start = js_sys::Date::now();

    let repr1 = get_repr(&repr1_str, (steps, n, n, m)).ok_or_else(|| JsValue::from_str(&format!("Failed to get representation 1, url: {}", repr1_str)))?;
    let repr2 = get_repr(&repr2_str, (steps, n, n, m)).ok_or_else(|| JsValue::from_str(&format!("Failed to get representation 2, url: {}", repr2_str)))?;
    let time_repr = js_sys::Date::now();

    let base_slice = repr1.slice(s![step1,row,col,..]);
    let mut similarities = Vec::with_capacity(n * n);
    let time_slice = js_sys::Date::now();

    for j in 0..n {
        for i in 0..n {
            let concept_slice = repr2.slice(s![step2,i,j,..]);
            let similarity = match func.as_str() {
                "cosine" => cosine_similarity(&base_slice, &concept_slice),
                "euclidean" => euclidean_distance(&base_slice, &concept_slice),
                "manhattan" => manhattan_distance(&base_slice, &concept_slice),
                "chebyshev" => chebyshev_distance(&base_slice, &concept_slice),
                _ => panic!("Unknown similarity function"),
            };
            similarities.push(similarity);
        }
    }
    let time_sim = js_sys::Date::now();

    if func == "euclidean" || func == "manhattan" || func == "chebyshev" {
        normalize_distances(&mut similarities);
    }
    let time_norm = js_sys::Date::now();

    console::log_1(&JsValue::from_str(&format!("Total {}, getting representations: {} ms, slicing: {} ms, similarity calculation: {} ms, normalization: {} ms", time_norm-time_start, time_repr - time_start, time_slice - time_repr, time_sim - time_slice, time_norm - time_sim)));

    Ok(similarities)
}

fn cosine_similarity(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    let dot_product = a * b;
    let norm_a = a.mapv(|x| x.powi(2)).sum().sqrt();
    let norm_b = b.mapv(|x| x.powi(2)).sum().sqrt();
    dot_product.sum() / (norm_a * norm_b)
}

fn normalize_distances(distances: &mut Vec<f32>) {
    let max_distance = *distances.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    for distance in distances.iter_mut() {
        *distance = 1.0 - (*distance / max_distance);
    }
}

fn euclidean_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

fn manhattan_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    (a - b).mapv(|x| x.abs()).sum()
}

fn chebyshev_distance(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    (a - b).mapv(|x| x.abs()).fold(0.0, |max, val| val.max(max))
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

    let float32_data_rc = float16_data.iter().map(|&x| f32::from(x)).collect::<Vec<f32>>();
    insert_repr(&url, float32_data_rc);

    Ok(())
}
