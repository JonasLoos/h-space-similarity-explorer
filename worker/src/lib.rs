use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use std::collections::HashMap;
use half::f16;
use web_sys::{js_sys, Request, RequestInit, RequestMode, Response};
// use web_sys::console
use std::cell::RefCell;
use js_sys::{ArrayBuffer, Uint8Array};
use console_error_panic_hook;
use std::panic;
use ndarray::{s, Array1, Array2, ArrayView1, Axis, Zip};
use std::sync::Arc;


macro_rules! jserr {() => {|e| JsValue::from_str(&format!("WASM: {:#?}", e))};}
macro_rules! jsnone {() => {JsValue::from_str(&format!("WASM: unexpected None in {}:{}:{}", file!(), line!(), column!()))};}


// cache to store representations and means
thread_local! {
    static GLOBAL_MAP: RefCell<HashMap<String, Arc<(Array2<f32>,Array1<f32>,Array1<f32>)>>> = RefCell::new(HashMap::new());
}


// calculate similarities between one pixel of a base representation and all pixels of a second representation
#[wasm_bindgen]
pub fn calc_similarities(
    func: String,
    repr1_str: String,
    repr2_str: String,
    row: usize,
    col: usize,
) -> Result<Vec<f32>, JsValue> {

    // better error messages in the console
    console_error_panic_hook::set_once();

    // log time for debugging
    // let time_start = js_sys::Date::now();

    // get representations from cache
    GLOBAL_MAP.with(|map| {
        let reprs = map.borrow();
        let arc_data1 = reprs.get(&repr1_str).ok_or_else(|| JsValue::from_str("loading"))?;
        let arc_data2 = reprs.get(&repr2_str).ok_or_else(|| JsValue::from_str("loading"))?;
        let (repr1, repr2, means1_full, means2_full, norms1, norms2) = (&arc_data1.0, &arc_data2.0, &arc_data1.1, &arc_data2.1, &arc_data1.2, &arc_data2.2);
        let n = (repr1.shape()[0] as f32).sqrt() as usize;

        // calculate mean of bath representations
        let means = if func == "cosine_centered" {
            Some(
                Zip::from(means1_full)
                    .and(means2_full)
                    .map_collect(|&mean1, &mean2| ((mean1 + mean2) / 2.0)))
        } else { None };

        // log time for debugging
        // let time_after_loading = js_sys::Date::now();

        // calculate similarities
        let a: ArrayView1<f32> = repr1.slice(s![row*n+col,..]);
        let mut similarities: Vec<f32> = match func.as_str() {
            "cosine" => repr2
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(index, b)| b.dot(&a) / (norms1[[row*n+col]] * norms2[[index]]))
                .collect::<Vec<_>>(),
            "cosine_centered" => repr2
                .axis_iter(Axis(0))
                .enumerate()
                .map(|(index, b)|
                    Zip::from(a)
                        .and(b)
                        .and(means.as_ref().unwrap().view())
                        .fold(0.0, |acc, &ai, &bi, &mean| acc + (ai - mean) * (bi - mean)) / (norms1[[row*n+col]] * norms2[[index]]))
                .collect::<Vec<_>>(),
            "manhattan" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).abs()))
                .collect::<Vec<_>>(),
            "euclidean" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc, &ai, &bi| acc + (ai - bi).powi(2)).sqrt())
                .collect::<Vec<_>>(),
            "chebyshev" => repr2
                .axis_iter(Axis(0))
                .map(|b| Zip::from(a).and(b).fold(0.0, |acc: f32, &ai, &bi| acc.max((ai - bi).abs())))
                .collect::<Vec<_>>(),
            _ => panic!("Unknown similarity function"),
        };

        // log time for debugging
        // let time_sim = js_sys::Date::now();

        // normalize distances
        if func == "euclidean" || func == "manhattan" || func == "chebyshev" {
            let max_distance = *similarities.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).ok_or(jsnone!())?;
            for distance in similarities.iter_mut() {
                *distance = 1.0 - (*distance / max_distance);
            }
        }

        // log time for debugging
        // let time_final = js_sys::Date::now();

        // log time for debugging
        // console::log_1(&JsValue::from_str(&format!("Total {}, getting representations: {} ms, similarity calculation: {} ms, normalization: {} ms", time_final-time_start, time_after_loading - time_start, time_sim - time_after_loading, time_final - time_sim)));

        Ok(similarities)
    })
}


// fetch representation from url and store it in cache
#[wasm_bindgen]
pub async fn fetch_repr(url: String, n: usize, m: usize) -> Result<(), JsValue> {
    console_error_panic_hook::set_once();  // better error messages in the console

    // if the representation is already fetched, return
    if GLOBAL_MAP.with(|map| map.borrow().contains_key(&url)) {
        return Ok(());
    }

    // initialize fetch request
    let mut opts = RequestInit::new();
    opts.method("GET");
    opts.mode(RequestMode::Cors);
    let request = Request::new_with_str_and_init(&url, &opts)?;

    // fetch representation
    let global = js_sys::global().unchecked_into::<web_sys::WorkerGlobalScope>();
    let resp: Response = match JsFuture::from(global.fetch_with_request(&request)).await {
        Ok(value) => value.dyn_into().map_err(jserr!())?,
        Err(e) => return Err(JsValue::from_str(&format!("Failed to fetch representation ({}): {:#?}", url, e)))
    };

    // convert response to float16 vector
    let buffer: ArrayBuffer = JsFuture::from(resp.array_buffer()?).await?.dyn_into().map_err(jserr!())?;
    if buffer.byte_length() % 2 != 0 {
        return Err(JsValue::from_str(&format!("Buffer length is not a multiple of 2 (for float16): {}", url)));
    }
    let bytes = Uint8Array::new(&buffer).to_vec();
    let float16_data: Vec<f16> = bytes.chunks_exact(2).map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]])).collect();

    // convert float16 vector to Array4<f32>
    let representations = match Array2::from_shape_vec((n*n, m), float16_data.iter().map(|&x| f32::from(x)).collect()) {
        Ok(repr) => repr,
        Err(e) => {
            return Err(JsValue::from_str(format!("Failed to convert float16 vector (len {}, {}) to Array3<f32> with shape ({}, {}): {:#?}", float16_data.len(), url, n*n, m, e).as_str()))
        }
    };

    // store representations and means and norms in cache
    let means = representations.mean_axis(Axis(0)).ok_or(jsnone!())?;
    let norms = representations.mapv(|x| x.powi(2)).sum_axis(Axis(1)).mapv(f32::sqrt);
    GLOBAL_MAP.with(|map| {
        map.borrow_mut().insert(url.to_string(), Arc::new((representations, means, norms)));
    });

    Ok(())
}
