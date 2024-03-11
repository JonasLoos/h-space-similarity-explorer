import * as wasm from './pkg/worker.js';

const asdf = wasm.default().then(_ => null);

onmessage = function(e) {
  asdf.then(_ => {
    const id = e.data.id;
    if (e.data.task === 'fetch_repr') {
      const { url, n, m } = e.data.data;
      wasm.fetch_repr(url, n, m)
        .then(() => postMessage({id, data: {status: 'success'}}))
        .catch((e) => postMessage({id, data: {status: 'error', msg: e}}))
    } else if (e.data.task === 'calc_similarities') {
      const { func, repr1_str, repr2_str, row, col } = e.data.data;
      try {
        const similarities = wasm.calc_similarities(func, repr1_str, repr2_str, row, col);
        postMessage({ id, data: similarities });
      } catch (e) {
        if (e === 'loading') {
          postMessage({ id, data: 'loading' });
        } else {
          console.error('error in calc_similarities\n', e);
          postMessage({ id, data: 'error' });
        }
      }
    }
  })
};
