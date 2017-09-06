extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct MeanSquaredError;

impl ops::Op for MeanSquaredError {
    fn name(&self) -> &str {
        "MeanSquaredError"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let a = xs[0];
        let b = xs[1];
        assert_eq!(a.shape(), b.shape());

        let mut diff = a - b;
        diff.mapv_inplace(|a| a * a * 0.5);
        diff
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let x0 = inputs[0];
        let x1 = inputs[1];
        let diff = x0 - x1;
        let gx2 = -1 * &diff;
        vec![Some(diff), Some(gx2)]
    }
}
