use ops;
use tensor::Tensor;

pub struct Identity;

impl ops::Op for Identity {
    fn name(&self) -> &str
    {
        "Identity"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        xs[0].clone()
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(gy.clone())]
    }
}
