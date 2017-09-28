use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct StopGradients;

impl ops::Op for StopGradients {
    fn name(&self) -> &str
    {
        "StopGradient"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].clone()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
