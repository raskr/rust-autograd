
use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct AddN;

impl ops::Op for AddN {
    fn name(&self) -> &str
    {
        "AddN"
    }

    fn compute(&mut self, xs: &[&::NdArray], _: bool) -> NdArray
    {
        let mut acc = NdArray::zeros(xs[0].shape());
        for &x in xs.iter() {
            acc += x;
        }
        acc
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        inputs
            .iter()
            .map(|_| Some((*gy).clone()))
            .collect::<Vec<Option<Tensor>>>()
    }
}
