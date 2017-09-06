extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Squeeze {
    pub axis: isize,
}

impl ops::Op for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let x = xs[0].clone();
        let axis = if self.axis == -1 {
            x.ndim() - 1
        } else {
            self.axis as usize
        };
        x.remove_axis(ndarray::Axis(axis))
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::expand_dims(gy, self.axis))]
    }
}
