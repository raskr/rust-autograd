extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct SparseSoftmaxCrossEntropy;
pub struct SparseSoftmaxCrossEntropyGrad;

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

        let log_x = ops::log_softmax::log_softmax_forward(xs[0], 1);

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

        let mut x = ops::softmax::softmax_forward(x, 1);

        for (mut row, t_) in x.axis_iter_mut(ndarray::Axis(0)).zip(t) {
            row[*t_ as usize] -= 1.;
        }

        x *= gy;
        x

    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
