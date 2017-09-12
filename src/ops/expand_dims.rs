extern crate ndarray;

use ops;
use tensor::Tensor;

pub struct ExpandDims {
    pub axes: Vec<isize>,
}


impl ops::Op for ExpandDims {
    fn name(&self) -> &str
    {
        "ExpandDims"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let ret = xs[0].clone();
        let mut output_shape = ret.shape().to_vec();
        for &i in self.axes.iter() {
            let axis = if i == -1 { ret.ndim() } else { i as usize };
            output_shape.insert(axis, 1);
        }
        ret.into_shape(output_shape).unwrap()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::squeeze(gy, self.axes.as_slice()))]
    }
}
