extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Squeeze;

impl ops::Op for Squeeze {
    fn name(&self) -> &str
    {
        "Squeeze"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let mut x = xs[0].view();
        let mut axes = xs[1].iter().map(|&a| a as isize).collect::<Vec<_>>();
        axes.sort();
        let mut adjust = 0;
        for &i in axes.iter() {
            let axis = if i < 0 {
                (x.ndim() as isize + i as isize) as usize
            } else {
                i as usize
            };
            let axis = axis - adjust;
            assert_eq!(
                1,
                x.shape()[axis],
                "Can't squeeze the dim whose length != 1"
            );
            // axis making ok
            x = x.remove_axis(ndarray::Axis(axis));
            adjust += 1;
        }
        x.to_owned()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::expand_dims(gy, inputs[1])), None]
    }
}
