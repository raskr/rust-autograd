use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Sigmoid;

impl ops::Op for Sigmoid {
    fn name(&self) -> &str
    {
        "Sigmoid"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let mut ret = x * 0.5;
        ret.mapv_inplace(|a| a.tanh());
        ret *= 0.5;
        ret += 0.5;
        ret
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some((output * (1 - output)) * gy)]
    }
}
