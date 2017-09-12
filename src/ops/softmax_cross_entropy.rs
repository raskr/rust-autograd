extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct SoftmaxCrossEntropy;

impl ops::Op for SoftmaxCrossEntropy {
    fn name(&self) -> &str
    {
        "SoftmaxCrossEntropy"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        // `t` must be one-hot unlike KL-divergence
        let x = xs[0];
        let t = xs[1];
        assert_eq!(x.ndim(), 2);
        assert_eq!(t.ndim(), 2);

        // - t log x ( =(batch, num_classes))
        let log_x = ops::log_softmax::log_softmax_forward(x, 1);
        // TODO: replace "sum" with "select"
        // unwrap is safe
        (t * &log_x).sum(ndarray::Axis(1)) * -1. // summing class dim.
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let ref x = ops::softmax(inputs[0], -1);
        let t = inputs[1];

        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = gy * (x - t);

        // gx2 won't be used
        let gx2 = {
            let log_x = ops::log_softmax(inputs[0], -1);
            let ref sum = ops::reduce_sum(&(x * &log_x), -1, true);
            gy * (sum - log_x) * output
        };

        vec![Some(gx1), Some(gx2)]
    }
}
