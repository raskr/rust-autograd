extern crate autograd as ag;
extern crate ndarray;

type Tensor = ag::Tensor<f32>;

struct LSTM {
    vector_dim: usize,
    hs: Vec<Tensor>,
    cells: Vec<Tensor>,
    wx: Tensor,
    wh: Tensor,
    b: Tensor,
}

impl LSTM {
    fn new(vector_dim: usize) -> LSTM {
        LSTM {
            vector_dim,
            hs: vec![],
            cells: vec![],
            wx: ag::variable(ag::ndarray_ext::random_normal(
                &[vector_dim, 4 * vector_dim],
                0.,
                0.01,
            )),
            wh: ag::variable(ag::ndarray_ext::random_normal(
                &[vector_dim, 4 * vector_dim],
                0.,
                0.01,
            )),
            b: ag::variable(ag::ndarray_ext::zeros(&[1, 4 * vector_dim])),
        }
    }

    /// Applies standard LSTM unit without peephole to `x`.
    /// `x` must be a tensor with shape `(batch_size, embedding_dim)`
    ///
    /// # Returns
    /// Output tensor of this unit with shape `(batch_size, state_size)`.
    fn step(&mut self, x: &Tensor) -> &Tensor {
        let (cell, h) = {
            let ref last_output = self.hs.pop().unwrap_or_else(|| ag::zeros(&x.shape()));
            let ref last_cell = self.cells.pop().unwrap_or_else(|| ag::zeros(&x.shape()));

            let ref xh = ag::matmul(x, &self.wx) + ag::matmul(last_output, &self.wh) + &self.b;

            let size = self.vector_dim as isize;
            let ref i = ag::slice(xh, &[0, 0 * size], &[-1, 1 * size]);
            let ref f = ag::slice(xh, &[0, 1 * size], &[-1, 2 * size]);
            let ref c = ag::slice(xh, &[0, 2 * size], &[-1, 3 * size]);
            let ref o = ag::slice(xh, &[0, 3 * size], &[-1, 4 * size]);

            let cell = ag::sigmoid(f) * last_cell + ag::sigmoid(i) * ag::tanh(c);
            let h = ag::sigmoid(o) * ag::tanh(&cell);
            (cell, h)
        };
        self.cells.push(cell);
        self.hs.push(h);
        self.hs.last().as_ref().unwrap()
    }
}

// TODO: Use real-world data
// TODO: Write in define-by-run style
pub fn main() {
    let vec_dim = 4;
    let max_sent = 2;
    let vocab_size = 5;

    let ref sentences = ag::placeholder(&[-1, max_sent]);
    let ref mut rnn = LSTM::new(vec_dim);

    let lookup_table = &ag::variable(ag::ndarray_ext::random_normal(
        &[vocab_size, vec_dim],
        0.,
        0.01,
    ));
    let w_pred = &ag::variable(ag::ndarray_ext::random_uniform(
        &[vec_dim, vocab_size],
        0.,
        0.01,
    ));

    // Compute cross entropy losses for each LSTM step
    let losses: Vec<Tensor> = (0..max_sent)
        .map(|i| {
            let cur_id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
            let next_id = ag::slice(sentences, &[0, i + 1], &[-1, i + 2]);
            let x = ag::gather(lookup_table, &cur_id, 0);
            let h = rnn.step(&x);
            let prediction = ag::matmul(h, w_pred);
            ag::sparse_softmax_cross_entropy(prediction, next_id)
        })
        .collect();

    // Aggregate losses of generated words
    let ref loss = ag::add_n(losses.iter().map(|a| a).collect::<Vec<_>>().as_slice());

    // Compute gradients
    let vars = &[&rnn.wh, &rnn.wx, &rnn.b, lookup_table, w_pred];
    let grads = ag::grad(&[loss], vars);

    // test with toy data
    ag::test_helper::check_theoretical_grads(
        loss,
        &grads,
        vars,
        &[(
            sentences,
            &ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]).into_dyn(),
        )],
        1e-3,
        1e-3,
    );
}
