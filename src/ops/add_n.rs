use ndarray_ext::NdArray;
use tensor::Tensor;
use ops;


pub struct AddN;

impl ops::Op for AddN {
    fn name(&self) -> &str
    {
        "AddN"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        if 0 == xs.len() {
            panic!("empty input to AddN")
        } else if 1 == xs.len() {
            xs[0].clone()
        } else {
            let mut base = xs[0] + xs[1];
            for &x in xs.iter() { base += x; }
            base
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        (0..inputs.len()).map(|_| Some(gy.clone())).collect::<Vec<Option<_>>>()
    }
}
