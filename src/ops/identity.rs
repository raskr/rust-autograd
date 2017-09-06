use tensor::Tensor;
use ops;
use ndarray_ext::NdArray;

pub struct Identity;

impl ops::Op for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    fn compute(&mut self, mut xs: &[&::NdArray], _: bool) -> ::NdArray {
        xs[0].clone()
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(gy.clone())]
    }
}
