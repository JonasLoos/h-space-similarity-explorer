import * as wasm from './pkg/worker.js';

onmessage = function(e) {
  wasm.default().then(_ => {
    const id = e.data.id;
    if (e.data.task === 'fetch_repr') {
      wasm.fetch_repr(e.data.data.url)
        .then(() => postMessage({id, data: {status: 'success'}}))
        .catch((e) => postMessage({id, data: {status: 'error', msg: e}}));
    } else if (e.data.task === 'calc_similarities') {
      const { func, repr1_str, repr2_str, step1, step2, row, col, steps, n, m } = e.data.data;
      try {
        const similarities = wasm.calc_similarities(func, repr1_str, repr2_str, step1, step2, row, col, steps, n, m);
        postMessage({ id, data: similarities });
      } catch (e) {
        // assume that the error is due to the loading of the representations
        console.log('error in calc_similarities, assuming representaitons are loading\n', e);
        postMessage({ id, data: 'loading' });
      }
    }
  })
};
