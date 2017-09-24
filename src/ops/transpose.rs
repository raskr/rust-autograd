use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Transpose;

impl ops::Op for Transpose {
    fn name(&self) -> &str
    {
        "Transpose"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let ndim = (&xs[0]).ndim() as isize;
        assert_eq!(ndim, 2);
        let mut ret: NdArray = xs[0].clone();
        ret.swap_axes(0, 1);
        ret
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::transpose(gy))]
    }
}
