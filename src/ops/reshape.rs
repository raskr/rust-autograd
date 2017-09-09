extern crate ndarray;

use std::mem;
use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;

pub struct Reshape {
    pub target_shape: Vec<usize>,
}

pub struct ReshapeGrad {
    pub target_shape: Vec<usize>,
}


impl ops::Op for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let ret = xs[0].clone();
        if let Ok(a) = ret.into_shape(ndarray::IxDyn(&*self.target_shape)) {
            a
        } else {
            panic!("reshape failed")
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let op = ReshapeGrad { target_shape: self.target_shape.clone() };
        vec![Some(ops::apply_op(op, &[inputs[0], gy]))]
    }
}

impl ops::Op for ReshapeGrad {
    fn name(&self) -> &str {
        "ReshapeGrad"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let x = xs[0];
        let gy: &NdArray = xs[1];
        let orig_shape = x.shape();
        // unwrap is safe
        gy.clone().into_shape(orig_shape).unwrap()
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let op = ReshapeGrad { target_shape: self.target_shape.clone() };
        let gy = inputs[1];
        vec![Some(ops::reshape(gy, self.target_shape.as_slice()))]
    }
}
