extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct Softmax {
    pub axis: isize,
}

#[inline]
pub fn softmax_forward(x: &NdArray, axis_: isize) -> NdArray
{
    let axis = if axis_ >= 0 {
        axis_ as usize
    } else {
        x.ndim() - 1
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
    tmp / &sum
}

impl ops::Op for Softmax {
    fn name(&self) -> &str
    {
        "Softmax"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        softmax_forward(xs[0], self.axis)
    }

    fn lop(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let sum = ops::reduce_sum(&(output * gy), self.axis, true);
        vec![Some((gy - sum) * output)]
    }
}
