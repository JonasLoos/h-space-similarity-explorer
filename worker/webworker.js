import * as wasm from './pkg/worker.js';

const asdf = wasm.default().then(_ => null);

onmessage = function(e) {
  asdf.then(_ => {
    const id = e.data.id;
    if (e.data.task === 'fetch_repr') {
      const { url, steps, n, m } = e.data.data;
      wasm.fetch_repr(url, steps, n, m)
        .then(() => postMessage({id, data: {status: 'success'}}))
        .catch((e) => postMessage({id, data: {status: 'error', msg: e}}))
        .finally(() => console.log('finished fetching repr'));
    } else if (e.data.task === 'calc_similarities') {
      const { func, repr1_str, repr2_str, step1, step2, row, col } = e.data.data;
      try {
        const similarities = wasm.calc_similarities(func, repr1_str, repr2_str, step1, step2, row, col);
        postMessage({ id, data: similarities });
      } catch (e) {
        // assume that the error is due to the loading of the representations
        console.log('error in calc_similarities, assuming representaitons are loading\n', e);
        postMessage({ id, data: 'loading' });
      }
    }
  })
};
