extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Squeeze {
    pub axes: Vec<isize>,
}

impl ops::Op for Squeeze {
    fn name(&self) -> &str {
        "Squeeze"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let mut x = xs[0].view();
        let mut adjust = 0;
        for &i in self.axes.iter() {
            let axis = if i == -1 { x.ndim() } else { i as usize };
            println!("remove: {}, {}, {}", x.ndim(), axis, axis-adjust);
            x = x.remove_axis(ndarray::Axis(axis-adjust));
            adjust += 1;
        }
        x.to_owned()
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::expand_dims(gy, self.axes.as_slice()))]
    }
}
