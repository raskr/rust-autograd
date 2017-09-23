extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct MatMul;

impl ops::Op for MatMul {
    fn name(&self) -> &str
    {
        "MatMul"
    }

    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let x0_shape = x0.shape();
        let x1_shape = x1.shape();
        assert_eq!(x0_shape.len(), 2);
        assert_eq!(x1_shape.len(), 2);
        let a = x0.view().into_shape((x0_shape[0], x0_shape[1])).unwrap();
        let b = x1.view().into_shape((x1_shape[0], x1_shape[1])).unwrap();
        a.dot(&b).into_dyn()
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let ga = ops::matmul(gy, &ops::swap_axes(inputs[1], 0, 1));
        let gb = ops::matmul(&ops::swap_axes(inputs[0], 0, 1), gy);
        vec![Some(ga), Some(gb)]
    }
}
