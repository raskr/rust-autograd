extern crate ndarray;

use std::mem;
use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;

pub struct Reshape {
    pub target_shape: Box<[usize]>,
    pub original_shape: Box<[usize]>,
}


impl ops::Op for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let ret = xs[0].clone();
        if 0 == self.original_shape.len() {
            mem::swap(
                &mut self.original_shape,
                &mut Box::new(ret.shape().to_vec().into_boxed_slice()),
            );
        }
        if let Ok(a) = ret.into_shape(ndarray::IxDyn(&*self.target_shape)) {
            a
        } else {
            panic!("reshape failed")
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![
            Some(ops::reshape(gy, self.original_shape.to_vec().as_slice())),
        ]
    }
}
