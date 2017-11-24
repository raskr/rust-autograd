extern crate ndarray;
extern crate autograd as ag;


struct LSTM {
    pub state_size: usize,
    pub batch_size: usize,
    pub last_output: ag::Tensor,
    pub cell: ag::Tensor,
    pub wx: ag::Tensor,
    pub wh: ag::Tensor,
    pub b: ag::Tensor,
}


impl LSTM {
    pub fn new(state_size: usize, input_dim: usize, batch_size: usize, c: &mut ag::Context)
        -> LSTM
    {
        LSTM {
            state_size,
            batch_size,
            last_output: ag::constant(ag::ndarray_ext::zeros(&[batch_size, state_size]), c),
            cell: ag::constant(ag::ndarray_ext::zeros(&[batch_size, state_size]), c),
            wx: ag::variable(
                ag::ndarray_ext::glorot_uniform(&[input_dim, 4 * state_size]),
                c,
            ),
            wh: ag::variable(
                ag::ndarray_ext::glorot_uniform(&[state_size, 4 * state_size]),
                c,
            ),
            b: ag::variable(ag::ndarray_ext::zeros(&[1, 4 * state_size]), c),
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
        let xh = ag::matmul(x, &self.wx) + ag::matmul(&self.last_output, &self.wh) + &self.b;

        let size = self.state_size as isize;
        let i = &ag::slice(&xh, &[0, 0 * size], &[-1, 1 * size]);
        let f = &ag::slice(&xh, &[0, 1 * size], &[-1, 2 * size]);
        let c = &ag::slice(&xh, &[0, 2 * size], &[-1, 3 * size]);
        let o = &ag::slice(&xh, &[0, 3 * size], &[-1, 4 * size]);

        let cell = ag::sigmoid(f) * &self.cell + ag::sigmoid(i) * ag::tanh(c);
        let h = ag::sigmoid(o) * ag::tanh(&cell);

        self.cell = cell;
        self.last_output = h.clone(); // data is not cloned
        h
    }

    fn reset_state(&mut self, c: &mut ag::Context)
    {
        self.last_output = ag::constant(
            ag::ndarray_ext::zeros(&[self.batch_size, self.state_size]),
            c,
        );

        self.cell = ag::constant(
            ag::ndarray_ext::zeros(&[self.batch_size, self.state_size]),
            c,
        );
    }
}

pub fn main()
{
    let state_size = 3;
    let vec_dim = 4;
    let max_sent = 2;
    let vocab_size = 5;
    let batch_size = 2;

    // === graph def
    let mut ctx = ag::Context::new();
    let ref tbl = ag::variable(
        ag::ndarray_ext::standard_uniform(&[vocab_size, vec_dim]),
        &mut ctx,
    );
    let ref w = ag::variable(
        ag::ndarray_ext::standard_normal(&[state_size, vocab_size]),
        &mut ctx,
    );
    let ref sentences = ag::placeholder(&[-1, max_sent]);
    let ref mut rnn = LSTM::new(state_size, vec_dim, batch_size, &mut ctx);

    let losses = (0..max_sent)
        .map(|i| {
            let ref cur_id = ag::slice(sentences, &[0, i], &[-1, i + 1]);
            let ref nex_id = ag::slice(sentences, &[0, i + 1], &[-1, i + 2]);
            let ref x = ag::gather(tbl, cur_id, 0);
            if i == max_sent - 1 {
                rnn.reset_state(&mut ctx);
            }
            let ref h = rnn.step(x);
            let ref prediction = ag::matmul(h, w);
            ag::sparse_softmax_cross_entropy(prediction, nex_id)
        })
        .collect::<Vec<_>>();

    let loss = losses.last().unwrap();
    let mut vars = rnn.list_vars();
    vars.extend_from_slice(&[tbl, w]);
    let ref g = ag::grad(&[loss], vars.as_slice());
    ctx.feed_input(sentences, ndarray::arr2(&[[2., 3., 1.], [3., 0., 1.]]));

    // test
    ag::test_helper::gradient_check(loss, g.as_slice(), vars.as_slice(), ctx, 1e-3, 1e-3);
}
