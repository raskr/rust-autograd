use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct ReverseAxes;

impl ops::Op for ReverseAxes {
    fn name(&self) -> &str
    {
        "ReverseAxes"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        xs[0].clone().reversed_axes()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::reverse_axes(gy))]
    }
}
