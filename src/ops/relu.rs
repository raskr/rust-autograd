use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;

pub struct ReLU;

impl ops::Op for ReLU {
    fn name(&self) -> &str {
        "ReLU"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        xs[0].map(|a| a.max(0.))
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::greater(inputs[0], 0.) * gy)]
    }
}
