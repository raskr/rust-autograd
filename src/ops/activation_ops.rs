extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use std::result::Result;
use tensor::Tensor;

pub struct ELU {
    pub alpha: f32,
}

pub struct ELUGrad {
    pub alpha: f32,
}

pub struct Identity;

pub struct ReLU;

pub struct Sigmoid;

pub struct Softplus;

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
    let sum = tmp.sum_axis(ndarray::Axis(axis))
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

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(softmax_forward(ctx.grab_inputs()[0], self.axis))
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let sum = ops::reduce_sum(&(output * gy), &[self.axis], true);
        vec![Some((gy - sum) * output)]
    }
}

impl ops::Op for Softplus {
    fn name(&self) -> &str
    {
        "Softplus"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let xs = ctx.grab_inputs();
        let e = f32::consts::E;
        Ok(xs[0].mapv(move |a| (a.exp() + 1.).log(e)))
    }

    fn grad(&self, gy: &Tensor, xs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let a = &ops::exp(xs[0]);
        let b = a + 1;
        let gx = gy * (a / b);
        vec![Some(gx)]
    }
}

impl ops::Op for Sigmoid {
    fn name(&self) -> &str
    {
        "Sigmoid"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = ctx.grab_inputs()[0];
        Ok(x.mapv(|a| ((a * 0.5).tanh() * 0.5) + 0.5))
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy * (y - ops::square(y)))]
    }
}

impl ops::Op for ReLU {
    fn name(&self) -> &str
    {
        "ReLU"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = ctx.grab_inputs()[0];
        Ok(x.map(|a| a.max(0.)))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let bin = ops::greater(inputs[0], &ops::scalar(0.));
        // inplace is ok because second derivative of relu is 0.
        // (`mul_inplace` returns `None` as input gradient.)
        vec![Some(ops::mul_inplace(bin, gy))]
    }
}

impl ops::Op for Identity {
    fn name(&self) -> &str
    {
        "Identity"
    }

    fn compute(&self, _: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        // do nothing
        Err(::OpComputeErrorStatus::Delegate { to: 0 })
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        // use gy's array with rc increment.
        vec![Some(gy.clone())]
    }
}

impl ops::Op for ELU {
    fn name(&self) -> &str
    {
        "ELU"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x = ctx.grab_inputs()[0];
        let ret = x.mapv(move |a| if a > 0. {
            a
        } else {
            self.alpha * (a.exp() - 1.)
        });
        Ok(ret)
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gx = Tensor::builder()
            .set_inputs(vec![inputs[0], gy])
            .set_shape(gy.shape())
            .build(ELUGrad { alpha: self.alpha });
        vec![Some(gx)]
    }
}

impl ops::Op for ELUGrad {
    fn name(&self) -> &str
    {
        "ELUGrad"
    }

    fn compute(&self, ctx: ::runtime::OpComputeContext) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let xs = ctx.grab_inputs();
        let x = xs[0];
        let gy = xs[1];
        let a = x.mapv(move |a| if a > 0. {
            1.
        } else {
            self.alpha * (a.exp() - 1.) + self.alpha
        });
        Ok(a * gy)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
