extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::f32;
use tensor::Tensor;


pub struct LogSumExp {
    pub axis: isize,
}


impl ops::Op for LogSumExp {
    fn name(&self) -> &str
    {
        "LogSumExp"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let axis = if self.axis >= 0 { self.axis as usize } else { x.ndim() - 1 };
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
        let mut sum = exp.sum(ndarray::Axis(axis))
                         .into_shape(ndarray::IxDyn(reduced_shape))
                         .unwrap();

        let e = f32::consts::E;
        sum.mapv_inplace(move |a| a.log(e));
        sum += max;
        sum
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        // let ref sum = ops::exp(output);
        // let ref exp = ops::exp(inputs[0]);
        // let gx = gy * exp / sum;
        let gx = ops::softmax(inputs[0], self.axis) * gy;
        vec![Some(gx)]
    }
}
