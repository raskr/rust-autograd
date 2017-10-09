extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;

pub struct ELU {
    pub alpha: f32,
}

pub struct ELUGrad {
    pub alpha: f32,
}

pub struct Identity;

pub struct ReLU;

pub struct ReLUGrad;

pub struct Sigmoid;

pub struct Softmax {
    pub axis: isize,
}

#[inline]
pub fn softmax_forward(x: &NdArray, axis: isize) -> NdArray
{
    let axis = if axis < 0 {
        (x.ndim() as isize + axis) as usize
    } else {
        axis as usize
    };

    let mut a = x.shape().to_vec();
    a[axis] = 1;
    let reduced_shape = a.as_slice();
    let max_fn = f32::max;
    // unwrap is safe
    let ref max = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    // subtract `max` to prevent overflow
    let mut tmp = x - max;
    tmp.mapv_inplace(|a| a.exp());
    // unwrap is safe
    let sum = tmp.sum(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();
    tmp /= &sum;
    tmp
}

impl ops::Op for Softmax {
    fn name(&self) -> &str
    {
        "Softmax"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        softmax_forward(xs[0], self.axis)
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let ref sum = ops::reduce_sum(&(output * gy), self.axis, true);

        vec![Some(ops::apply_op(ops::binary_ops::InplaceSubOp, &[gy, sum]) * output)]
    }
}

impl ops::Op for Sigmoid {
    fn name(&self) -> &str
    {
        "Sigmoid"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].mapv(|a| ((a*0.5).tanh() * 0.5) + 0.5)
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some((output * (1 - output)) * gy)]
    }
}

impl ops::Op for ReLU {
    fn name(&self) -> &str
    {
        "ReLU"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].map(|a| a.max(0.))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::apply_op(ReLUGrad, &[inputs[0], gy]))]
    }
}

impl ops::Op for ReLUGrad {
    fn name(&self) -> &str
    {
        "ReLUGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let mut bin = xs[0].mapv(move |a| ((a > 0.) as i32) as f32);
        bin *= xs[1];
        bin
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Identity {
    fn name(&self) -> &str
    {
        "Identity"
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].clone()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone())]
    }
}

impl ops::Op for ELU {
    fn name(&self) -> &str
    {
        "ELU"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].mapv(move |a| if a > 0. {
            a
        } else {
            self.alpha * (a.exp() - 1.)
        })
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gx = ops::apply_op(ELUGrad { alpha: self.alpha }, &[inputs[0], gy]);
        vec![Some(gx)]
    }
}

impl ops::Op for ELUGrad {
    fn name(&self) -> &str
    {
        "ELUGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy = xs[1];
        let a = x.mapv(move |a| if a > 0. {
            1.
        } else {
            self.alpha * (a.exp() - 1.) + self.alpha
        });
        a * gy
    }

    // TODO: impl
    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
