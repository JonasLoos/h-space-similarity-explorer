use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use std::collections::HashMap;
use half::f16;
use web_sys::{js_sys, Request, RequestInit, RequestMode, Response, console};
use std::cell::RefCell;
use js_sys::{ArrayBuffer, Uint8Array};
use console_error_panic_hook;
use std::panic;
use ndarray::{s, Array2, Array3, ArrayView1, Axis, Zip};
use std::sync::Arc;


// cache to store representations and means
thread_local! {
    static GLOBAL_MAP: RefCell<HashMap<String, Arc<(Array3<f32>,Array2<f32>,Array2<f32>)>>> = RefCell::new(HashMap::new());
}


// calculate similarities between one pixel of a base representation and all pixels of a second representation
#[wasm_bindgen]
pub fn calc_similarities(
    func: String,
    repr1_str: String,
    repr2_str: String,
    step1: usize,
    step2: usize,
    row: usize,
    col: usize,
) -> Result<Vec<f32>, JsValue> {

    // better error messages in the console
    console_error_panic_hook::set_once();

    // log time for debugging
    let time_start = js_sys::Date::now();

    // get representations from cache
    GLOBAL_MAP.with(|map| {
        let reprs = map.borrow();
        let arc_data1 = reprs.get(&repr1_str).ok_or_else(|| JsValue::from_str(&format!("Failed to get representation, url: {}", repr1_str)))?;
        let arc_data2 = reprs.get(&repr2_str).ok_or_else(|| JsValue::from_str(&format!("Failed to get representation, url: {}", repr2_str)))?;
        let (repr1, repr2, means1_full, means2_full, norms1, norms2) = (&arc_data1.0, &arc_data2.0, &arc_data1.1, &arc_data2.1, &arc_data1.2, &arc_data2.2);
        let n = (repr1.shape()[1] as f32).sqrt() as usize;

        // calculate mean of bath representations
        let means = if func == "cosine_centered" {
            Some(
                Zip::from(&means1_full.slice(s![step1,..]))
                    .and(&means2_full.slice(s![step2,..]))
                    .map_collect(|&mean1, &mean2| ((mean1 + mean2) / 2.0)))
        } else { None };

        // log time for debugging
        let time_after_loading = js_sys::Date::now();

        // calculate similarities
        let a: ArrayView1<f32> = repr1.slice(s![step1,row*n+col,..]);
        let mut similarities: Vec<f32> = match func.as_str() {
            "cosine" => repr2.slice(s![step2, .., ..])
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(index, b)| b.dot(&a) / (norms1[[step1, index]] * norms2[[step2, index]]))
                .collect::<Vec<_>>(),
            "cosine_centered" => repr2.slice(s![step2,..,..])
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(index, b)|
                    Zip::from(a)
                        .and(b)
                        .and(means.as_ref().unwrap().view())
                        .fold(0.0, |acc, &ai, &bi, &mean| acc + (ai - mean) * (bi - mean)) / (norms1[[step1, index]] * norms2[[step2, index]]))
                .collect::<Vec<_>>(),
            "manhattan" => repr2.slice(s![step2,..,..])
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).abs()))
                .collect::<Vec<_>>(),
            "euclidean" => repr2.slice(s![step2,..,..])
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).powi(2)).sqrt())
                .collect::<Vec<_>>(),
            "chebyshev" => repr2.slice(s![step2,..,..])
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc: f32, &ai, &bi| acc.max((ai - bi).abs())))
                .collect::<Vec<_>>(),
            _ => panic!("Unknown similarity function"),
        };

        // log time for debugging
        let time_sim = js_sys::Date::now();

        // normalize distances
        if func == "euclidean" || func == "manhattan" || func == "chebyshev" {
            let max_distance = *similarities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            for distance in similarities.iter_mut() {
                *distance = 1.0 - (*distance / max_distance);
            }
        }

        // log time for debugging
        let time_final = js_sys::Date::now();

        // log time for debugging
        console::log_1(&JsValue::from_str(&format!("Total {}, getting representations: {} ms, similarity calculation: {} ms, normalization: {} ms", time_final-time_start, time_after_loading - time_start, time_sim - time_after_loading, time_final - time_sim)));

        Ok(similarities)
    })
}


// fetch representation from url and store it in cache
#[wasm_bindgen]
pub async fn fetch_repr(url: String, steps: usize, n: usize, m: usize) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();  // better error messages in the console

    // if the representation is already fetched, return
    if GLOBAL_MAP.with(|map| map.borrow().contains_key(&url)) {
        console::log_1(&JsValue::from_str(&format!("WASM: Representation already fetched: {}", url)));
        return Ok(());
    }

    // initialize fetch request
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(&url, &opts)?;

    // fetch representation
    let global = js_sys::global().unchecked_into::<web_sys::WorkerGlobalScope>();
    let resp_value = match JsFuture::from(global.fetch_with_request(&request)).await {
        Ok(value) => value,
        Err(e) => {
            console::warn_1(&JsValue::from_str(&format!("WASM: Fetch error: {:?}", e)));
            return Err(JsValue::from_str(&format!("Failed to fetch representation: {}", url)))
        }
    };
    let resp: Response = match resp_value.dyn_into() {
        Ok(resp) => resp,
        Err(_) => {
            console::warn_1(&JsValue::from_str("WASM: Failed to get Response: resp_value.dyn_into() failed"));
            return Err(JsValue::from_str("Failed to get Response: resp_value.dyn_into() failed"));
        }
    };

    // convert response to float16 vector
    let buffer_value = JsFuture::from(resp.array_buffer()?).await?;
    let buffer: ArrayBuffer = match buffer_value.dyn_into() {
        Ok(buffer) => buffer,
        Err(_) => return Err(JsValue::from_str("Failed to get ArrayBuffer: buffer_value.dyn_into() failed")),
    };
    if buffer.byte_length() % 2 != 0 {
        return Err(JsValue::from_str("Buffer length is not a multiple of 2 (for float16)"));
    }
    let bytes = Uint8Array::new(&buffer).to_vec();
    let float16_data: Vec<f16> = bytes.chunks(2).map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]])).collect();

    // convert float16 vector to Array4<f32> and store it in cache
    let representations = Array3::from_shape_vec((steps, n*n, m), float16_data.iter().map(|&x| f32::from(x)).collect()).unwrap();
    let means = representations.mean_axis(Axis(1)).unwrap();
    let norms = representations.mapv(|x| x.powi(2)).sum_axis(Axis(2)).mapv(f32::sqrt);
    GLOBAL_MAP.with(|map| {
        map.borrow_mut().insert(url.to_string(), Arc::new((representations, means, norms)));
    });

    Ok(())
}
