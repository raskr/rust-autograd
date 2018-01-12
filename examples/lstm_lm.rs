extern crate ndarray;
extern crate autograd as ag;


struct LSTM {
    pub vector_dim: usize,
    pub hs: Vec<ag::Tensor>,
    pub cells: Vec<ag::Tensor>,
    pub wx: ag::Tensor,
    pub wh: ag::Tensor,
    pub b: ag::Tensor,
}


impl LSTM {
    pub fn new(vector_dim: usize) -> LSTM
    {
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

    pub fn list_vars(&self) -> Vec<&ag::Tensor>
    {
        vec![&self.wx, &self.wh, &self.b]
    }

    #[inline]
    /// Applies standard LSTM unit without peephole to input.
    ///
    /// # Arguments
    /// * `x` - Tensor with shape (batch_size, embedding_dim)
    ///
    /// # Returns
    /// output tensor of this unit with shape (batch_size, state_size)
    fn step(&mut self, x: &ag::Tensor) -> ag::Tensor
    {
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
        self.hs.push(h.clone());
        h
    }
}

pub fn main()
{
    let vec_dim = 4;
    let max_sent = 2;
    let vocab_size = 5;

    // === graph def
    let ref sentences = ag::placeholder(&[-1, max_sent]);
    let ref mut rnn = LSTM::new(vec_dim);

    let ref lookup_table = ag::variable(ag::ndarray_ext::random_normal(
        &[vocab_size, vec_dim],
        0.,
        0.01,
    ));
    let ref w_pred = ag::variable(ag::ndarray_ext::random_uniform(
        &[vec_dim, vocab_size],
        0.,
        0.01,
    ));
    // Compute cross entropy losses for each LSTM step
    let losses = (0..max_sent)
        .map(|i| {
            let ref cur_id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
            let ref next_id = ag::slice(sentences, &[0, i + 1], &[-1, i + 2]);
            let ref x = ag::gather(lookup_table, cur_id, 0);
            let ref h = rnn.step(x);
            let ref prediction = ag::matmul(h, w_pred);
            ag::sparse_softmax_cross_entropy(prediction, next_id)
        })
        .collect::<Vec<_>>();

    // For simplicity, I use loss of the last output only.
    let loss = losses.last().unwrap();
    let mut vars = rnn.list_vars();
    vars.extend_from_slice(&[lookup_table, w_pred]);
    let ref g = ag::grad(&[loss], vars.as_slice());
    // === graph def end

    // test with toy data
    ag::test_helper::gradient_check(
        loss,
        g.as_slice(),
        vars.as_slice(),
        &[
            (
                sentences,
                &ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]).into_dyn(),
            ),
        ],
        1e-3,
        1e-3,
    );
}
