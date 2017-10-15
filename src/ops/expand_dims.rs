extern crate ndarray;

use ops;
use tensor::Tensor;

pub struct ExpandDims;


impl ops::Op for ExpandDims {
    fn name(&self) -> &str
    {
        "ExpandDims"
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let ret = xs[0].clone();
        let mut axes = xs[1].iter().map(|&a| a as isize).collect::<Vec<_>>();
        axes.sort();
        let mut output_shape = ret.shape().to_vec();
        for &i in axes.iter() {
            let axis = if i < 0 {
                (ret.ndim() as isize + i as isize) as usize
            } else {
                i as usize
            };
            output_shape.insert(axis, 1);
        }
        ret.into_shape(output_shape).unwrap()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::squeeze(gy, inputs[1])), None]
    }
}
