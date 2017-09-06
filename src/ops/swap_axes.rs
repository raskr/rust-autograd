use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct SwapAxes {
    pub a: isize,
    pub b: isize,
}


impl ops::Op for SwapAxes {
    fn name(&self) -> &str {
        "SwapAxes"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray {
        let ndim = (&xs[0]).ndim();
        let a = if self.a == -1 { ndim } else { self.a as usize };
        let b = if self.b == -1 { ndim } else { self.b as usize };
        let mut ret: NdArray = xs[0].clone();
        ret.swap_axes(a, b);
        ret
    }

    fn lop(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>> {
        vec![Some(ops::swap_axes(gy, self.a, self.b))]
    }
}
