import * as wasm from './pkg/worker.js';

onmessage = function(e) {
  wasm.default().then(_ => {
    if (e.data.task === 'fetch_repr') {
      wasm.fetch_repr(e.data.data.url).then(() => postMessage('success')).catch(() => postMessage('failure'));
    } else if (e.data.task === 'calc_similarities') {
      const { func, repr1_str, repr2_str, step1, step2, row, col, steps, n, m } = e.data.data;
      try {
        const similarities = wasm.calc_similarities(func, repr1_str, repr2_str, step1, step2, row, col, steps, n, m);
        postMessage({ similarities });
      } catch (e) {
        postMessage({ similarities: 'loading' });
      }
    }
  })
};
