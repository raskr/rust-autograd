extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct SigmoidCrossEntropy;

impl ops::Op for SigmoidCrossEntropy {
    fn name(&self) -> &str
    {
        "SigmoidCrossEntropy"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let t = xs[1];

        assert!(x.shape() == t.shape());

        let e = f32::consts::E;
        let max_fn = f32::max;
        x.map(move |a| ((-a.abs()).exp() + 1.).log(e) + max_fn(0., *a)) - t * x
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let t = inputs[1];

        let gx1 = {
            let ref exp = ops::exp(x);
            ((exp / (exp + 1)) - t) * gy
        };

        let gx2 = -1 * gy * t;

        vec![Some(gx1), Some(gx2)]
    }
}
