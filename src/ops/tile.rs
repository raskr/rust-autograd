extern crate ndarray;

use tensor::Tensor;
use ndarray_ext::NdArray;
use ops;


pub struct Tile {
    pub axis: isize,
    pub num: usize,
}

impl ops::Op for Tile {

    fn name(&self) -> &str { "Tile" }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x: &NdArray = xs[0];

        let axis = if self.axis >= 0 {
            self.axis as usize
        } else {
            x.ndim() - 1
        };

        let mut views = vec![];
        for i in 0..self.num {
            views.push(x.view());
        }
        // TODO: remove unwrap
        ndarray::stack(ndarray::Axis(axis), views.as_slice()).unwrap()
    }

    fn lop(&self, gy: &Tensor,
           inputs: &[&Tensor],
           output: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![Some(ops::reduce_sum(gy, self.axis, true))]
    }

}
