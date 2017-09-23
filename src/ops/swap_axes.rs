use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct SwapAxes {
    pub a: isize,
    pub b: isize,
}


impl ops::Op for SwapAxes {
    fn name(&self) -> &str
    {
        "SwapAxes"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let ndim = (&xs[0]).ndim() as isize;
        let a = if self.a < 0 { (ndim + self.a) as usize } else { self.a as usize };
        let b = if self.b < 0 { (ndim + self.b) as usize } else { self.b as usize };
        let mut ret: NdArray = xs[0].clone();
        ret.swap_axes(a, b);
        ret
    }

    fn grad(&self, gy: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::swap_axes(gy, self.a, self.b))]
    }
}
