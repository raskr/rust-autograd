extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct LogSoftmax {
    pub axis: isize,
}

pub fn logsumexp(x: &NdArray, axis: isize) -> NdArray
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
    let ref max = x.fold_axis(ndarray::Axis(axis), f32::MIN, move |&a, &b| max_fn(a, b))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    // subtract `max` to prevent overflow of exp
    let mut tmp = x - max;

    let exp = {
        tmp.mapv_inplace(|a| a.exp());
        tmp
    };

    // unwrap is safe
    let mut sum = exp.sum_axis(ndarray::Axis(axis))
        .into_shape(ndarray::IxDyn(reduced_shape))
        .unwrap();

    let e = f32::consts::E;
    sum.mapv_inplace(move |a| a.log(e));
    sum += max;
    sum
}

pub fn log_softmax_forward(x: &NdArray, axis: isize) -> NdArray
{
    x - &logsumexp(x, axis)
}

impl ops::Op for LogSoftmax {
    fn name(&self) -> &str
    {
        "LogSoftmax"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        log_softmax_forward(x, self.axis)
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let sm = ops::exp(output);
        let sum = ops::reduce_sum(gy, 1, true);
        let ref mul = sm * sum;
        vec![Some(gy - mul)]
    }
}
