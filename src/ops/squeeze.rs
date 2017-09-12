extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Squeeze {
    pub axes: Vec<isize>,
}

impl ops::Op for Squeeze {
    fn name(&self) -> &str
    {
        "Squeeze"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let mut x = xs[0].view();
        let mut adjust = 0;
        for &i in self.axes.iter() {
            let axis = if i == -1 { x.ndim() } else { i as usize };
            let axis = axis - adjust;
            assert_eq!(1, x.shape()[axis], "Can't squeeze the dim whose length != 1");
            x = x.remove_axis(ndarray::Axis(axis));
            adjust += 1;
        }
        x.to_owned()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::expand_dims(gy, self.axes.as_slice()))]
    }
}
