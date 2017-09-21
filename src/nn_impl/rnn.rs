extern crate ndarray;

use initializers;
use ops;
use tensor::Tensor;


pub trait RNN {
    fn step(&mut self, x: &Tensor) -> Tensor;
    fn reset_state(&mut self);
}


/// Standard LSTM unit without peephole.
pub struct LSTM {
    pub state_size: usize,
    pub batch_size: usize,
    pub last_output: Tensor,
    pub cell: Tensor,
    pub wx: Tensor,
    pub wh: Tensor,
    pub b: Tensor,
}

impl LSTM {
    #[inline]
    pub fn new(state_size: usize, input_dim: usize, batch_size: usize) -> LSTM
    {
        LSTM {
            state_size: state_size,
            batch_size: batch_size,
            last_output: ops::constant(initializers::zeros(&[batch_size, state_size])),
            cell: ops::constant(initializers::zeros(&[batch_size, state_size])),
            wx: ops::variable(initializers::glorot_uniform(&[input_dim, 4 * state_size])),
            wh: ops::variable(initializers::glorot_uniform(&[state_size, 4 * state_size])),
            b: ops::variable(initializers::zeros(&[1, 4 * state_size])),
        }
    }

    #[inline]
    pub fn list_vars(&self) -> Vec<&Tensor>
    {
        vec![&self.wx, &self.wh, &self.b]
    }

    #[inline]
    pub fn get_current_cell(&self) -> &Tensor
    {
        &self.cell
    }
}

impl RNN for LSTM {
    /// Applies standard LSTM unit without peephole to input.
    ///
    /// # Arguments
    /// * `x` - Tensor with shape (batch_size, embedding_dim)
    ///
    /// # Returns
    /// output tensor of this unit with shape (batch_size, state_size)
    #[inline]
    fn step(&mut self, x: &Tensor) -> Tensor
    {
        let xh = ops::matmul(x, &self.wx) + ops::matmul(&self.last_output, &self.wh) + &self.b;

        let size = self.state_size as isize;
        let i = &ops::slice(&xh, &[0, 0 * size], &[-1, 1 * size]);
        let f = &ops::slice(&xh, &[0, 1 * size], &[-1, 2 * size]);
        let c = &ops::slice(&xh, &[0, 2 * size], &[-1, 3 * size]);
        let o = &ops::slice(&xh, &[0, 3 * size], &[-1, 4 * size]);

        let cell = ops::sigmoid(f) * &self.cell + ops::sigmoid(i) * ops::tanh(c);
        let h = ops::sigmoid(o) * ops::tanh(&cell);

        self.cell = cell;
        self.last_output = h.clone(); // data is not cloned
        h
    }

    #[inline]
    fn reset_state(&mut self)
    {
        self.last_output = ops::constant(initializers::zeros(&[self.batch_size, self.state_size]));

        self.cell = ops::constant(initializers::zeros(&[self.batch_size, self.state_size]));
    }
}
