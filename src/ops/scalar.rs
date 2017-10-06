extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Scalar {
    pub val: f32,
}

impl ops::Op for Scalar {
    fn name(&self) -> &str
    {
        "Scalar"
    }

    fn compute(&self, _: &[&NdArray], _: bool) -> NdArray
    {
        NdArray::from_elem(ndarray::IxDyn(&[1]), self.val)
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
