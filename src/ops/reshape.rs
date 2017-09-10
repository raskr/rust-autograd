extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Reshape {
    pub target_shape: Vec<Option<usize>>,
}

pub struct ReshapeGrad {
    pub target_shape: Vec<Option<usize>>,
}


impl ops::Op for Reshape {
    fn name(&self) -> &str {
        "Reshape"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let ret = xs[0].clone();

        let target = self.target_shape.iter().map(|opt| {
            if let &Some(len) = opt {
                len
            } else {
                let a = self.target_shape.iter().fold(1, |acc, x| acc * x.unwrap_or(1));
                ret.len() - if a == 1 { 0 } else { a }
            }
        }).collect::<Vec<_>>();

        if let Ok(a) = ret.into_shape(ndarray::IxDyn(target.as_slice())) {
            a
        } else {
            panic!("reshape failed")
        }
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        let op = ReshapeGrad {
            target_shape: self.target_shape.clone(),
        };
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
        let op = ReshapeGrad {
            target_shape: self.target_shape.clone(),
        };
        let gy = inputs[1];
        let reshape = Reshape {
            target_shape: self.target_shape.clone()
        };
        vec![None, Some(ops::apply_op(reshape, &[gy]))]
    }
}
