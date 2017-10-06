use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;

pub struct ReLU;

impl ops::Op for ReLU {
    fn name(&self) -> &str
    {
        "ReLU"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].map(|a| a.max(0.))
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::greater(inputs[0], 0.) * gy)]
    }
}
