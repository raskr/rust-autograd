extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct SoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropyGrad;
pub struct SigmoidCrossEntropy;

impl ops::Op for SigmoidCrossEntropy {
    fn name(&self) -> &str
    {
        "SigmoidCrossEntropy"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let t = xs[1];

        assert_eq!(x.shape(), t.shape());

        let e = f32::consts::E;
        let max_fn = f32::max;
        let mut tmp = x.map(move |a| ((-a.abs()).exp() + 1.).log(e) + max_fn(0., *a));
        tmp -= &(t * x);
        tmp
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let t = inputs[1];

        let gx1 = {
            let ref exp = ops::exp(x);
            ((exp / (exp + 1)) - t) * gy
        };

        let gx2 = -1 * gy * t;

        vec![Some(gx1), Some(gx2)]
    }
}

impl ops::Op for SparseSoftmaxCrossEntropy {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropy"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let (x, t) = (xs[0], xs[1]);
        {
            assert_eq!(x.ndim(), 2);
            let t_ndim = t.ndim();
            assert!(1 == t_ndim || 2 == t_ndim);
        }

        let mut log_x = ops::log_softmax::log_softmax_forward(xs[0], 1);

        let mut t_iter = t.iter();

        // unwrap is safe
        let ret = log_x
            .map_axis(ndarray::Axis(1), move |row| {
                -row[*t_iter.next().unwrap() as usize]
            })
            .into_shape(ndarray::IxDyn(&[log_x.shape()[0], 1]))
            .unwrap();

        ret
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let x = inputs[0];
        let t = inputs[1];

        let gx1 = ops::apply_op(SparseSoftmaxCrossEntropyGrad, &[x, t, gy]);

        // gx2 won't be used
        let gx2 = {
            let ref log_x = ops::log_softmax(inputs[0], -1);
            let ref x = ops::exp(log_x);
            let sum = ops::reduce_sum(&(x * log_x), 1, true);
            x * gy * (sum - log_x)
        };

        vec![Some(gx1), Some(gx2)]
    }
}

impl ops::Op for SparseSoftmaxCrossEntropyGrad {
    fn name(&self) -> &str
    {
        "SparseSoftmaxCrossEntropyGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let t = xs[1];
        let gy = xs[2];

        let mut x = ops::activation_ops::softmax_forward(x, 1);

        for (mut row, &t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[t_ as usize] -= 1.;
        }

        x *= gy;
        x

    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
impl ops::Op for SoftmaxCrossEntropy {
    fn name(&self) -> &str
    {
        "SoftmaxCrossEntropy"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        // `t` must be one-hot unlike KL-divergence
        let x = xs[0];
        let t = xs[1];
        assert_eq!(x.ndim(), 2);
        assert_eq!(t.ndim(), 2);

        // - t log x ( =(batch, num_classes))
        let log_x = ops::log_softmax::log_softmax_forward(x, 1);
        // TODO: replace "sum" with "select"
        // unwrap is safe
        (t * &log_x).sum(ndarray::Axis(1)) * -1. // summing class dim.
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        let ref x = ops::softmax(inputs[0], -1);
        let t = inputs[1];

        // = {gy - Σ(x * gy)} * x
        // = {-t/x - Σ(x * -t/x)} * x
        // = {-t/x + Σt} * x
        // = -t + x
        let gx1 = ops::apply_op(ops::binary_ops::InplaceSubOp, &[x, t]) * gy;

        // gx2 won't be used
        let gx2 = {
            let ref log_x = ops::log_softmax(inputs[0], -1);
            let sum = ops::reduce_sum(&(x * log_x), -1, true);
            gy * (sum - log_x) * output
        };

        vec![Some(gx1), Some(gx2)]
    }
}
