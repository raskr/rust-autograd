extern crate ndarray;

use std::mem;
use tensor::Tensor;
use ops;
use ndarray_ext::NdArray;

pub struct ExpandDims {
    pub axis: isize,
}


impl ops::Op for ExpandDims {
    fn name(&self) -> &str {
        "ExpandDims"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> ::NdArray {
        let ret = xs[0].clone();
        let axis = if self.axis == -1 {
            ret.ndim()
        } else {
            self.axis as usize
        };
        let mut output_shape = ret.shape().to_vec();
        output_shape.insert(axis, 1);
        ret.into_shape(output_shape).unwrap()
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::squeeze(gy, self.axis))]
    }
}
