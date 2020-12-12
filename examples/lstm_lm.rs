extern crate autograd as ag;
extern crate ndarray;

use ag::tensor::Variable;
use ag::Graph;

type Tensor<'graph> = ag::Tensor<'graph, f32>;

struct LSTM<'g> {
    vector_dim: usize,
    hs: Vec<Tensor<'g>>,
    cells: Vec<Tensor<'g>>,
    wx: Tensor<'g>,
    wh: Tensor<'g>,
    b: Tensor<'g>,
}

impl<'g> LSTM<'g> {
    fn new(vector_dim: usize, s: &'g Graph<f32>) -> LSTM<'g> {
        let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
        LSTM {
            vector_dim,
            hs: vec![],
            cells: vec![],
            wx: s.variable(rng.random_normal(&[vector_dim, 4 * vector_dim], 0., 0.01)),
            wh: s.variable(rng.random_normal(&[vector_dim, 4 * vector_dim], 0., 0.01)),
            b: s.variable(ag::ndarray_ext::zeros(&[1, 4 * vector_dim])),
        }
    }

    /// Applies standard LSTM unit without peephole to `x`.
    /// `x` must be a tensor with shape `(batch_size, embedding_dim)`
    ///
    /// # Returns
    /// Output tensor of this unit with shape `(batch_size, state_size)`.
    fn step(&mut self, x: Tensor<'g>, s: &'g Graph<f32>) -> &Tensor<'g> {
        let (cell, h) = {
            let ref last_output = self.hs.pop().unwrap_or_else(|| s.zeros(&s.shape(x)));
            let last_cell = self.cells.pop().unwrap_or_else(|| s.zeros(&s.shape(x)));

            let xh = s.matmul(x, self.wx) + s.matmul(last_output, self.wh) + self.b;

            let size = self.vector_dim as isize;
            let i = s.slice(xh, &[0, 0 * size], &[-1, 1 * size]);
            let f = s.slice(xh, &[0, 1 * size], &[-1, 2 * size]);
            let c = s.slice(xh, &[0, 2 * size], &[-1, 3 * size]);
            let o = s.slice(xh, &[0, 3 * size], &[-1, 4 * size]);

            let cell = s.sigmoid(f) * last_cell + s.sigmoid(i) * s.tanh(c);
            let h = s.sigmoid(o) * s.tanh(&cell);
            (cell, h)
        };
        self.cells.push(cell);
        self.hs.push(h);
        self.hs.last().unwrap()
    }
}

// TODO: Use real-world data
// TODO: Write in define-by-run style
pub fn main() {
    let vec_dim = 4;
    let max_sent = 3;
    let vocab_size = 5;

    ag::with(|s| {
        let sentences = s.placeholder(&[-1, max_sent]);
        let ref mut rnn = LSTM::new(vec_dim, s);

        let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
        let lookup_table = s.variable(rng.random_normal(&[vocab_size, vec_dim], 0., 0.01));
        let w_pred = s.variable(rng.random_uniform(&[vec_dim, vocab_size], 0., 0.01));

        // Compute cross entropy losses for each LSTM step
        let losses: Vec<ag::Tensor<_>> = (0..max_sent - 1)
            .map(|i| {
                let cur_id = s.slice(sentences, &[0, i], &[-1, i + 1]);
                let next_id = s.slice(sentences, &[0, i + 1], &[-1, i + 2]);
                let x = s.squeeze(s.gather(lookup_table, &cur_id, 0), &[1]);
                let h = rnn.step(x, s);
                let prediction = s.matmul(h, w_pred);
                s.sparse_softmax_cross_entropy(prediction, next_id)
            })
            .collect();

        // Aggregate losses of generated words
        let loss = s.add_n(&losses);

        // Compute gradients
        let vars = &[rnn.wh, rnn.wx, rnn.b, lookup_table, w_pred];
        let grads = s.grad(&[loss], vars);

        // test with toy data
        ag::test_helper::check_theoretical_grads(
            loss,
            grads.as_slice(),
            vars,
            &[sentences.given(
                ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]])
                    .into_dyn()
                    .view(),
            )],
            1e-3,
            1e-3,
        );
    });
}
