use std::mem;
use tensor::{Tensor, RawTensor};
use ops;
use ndarray_ext::NdArray;


pub struct AddN;

impl ops::Op for AddN {
    fn name(&self) -> &str {
        "AddN"
    }

    fn compute(&mut self, xs: &[&::NdArray], train: bool) -> NdArray {
        let mut acc = NdArray::zeros(xs[0].shape());
        for &x in xs.iter() {
            acc += x;
        }
        acc
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        inputs.iter().map(|_| Some((*gy).clone())).collect::<Vec<Option<Tensor>>>()
    }
}
