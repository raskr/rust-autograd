extern crate ndarray;

use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use std::mem;
use tensor::Tensor;


pub struct MatMul;

impl ops::Op for MatMul {
    fn name(&self) -> &str
    {
        "MatMul"
    }

    #[allow(mutable_transmutes)]
    fn compute(&mut self, xs: &[&NdArray], _: bool) -> NdArray
    {
        assert_eq!(xs[0].ndim(), 2);
        assert_eq!(xs[1].ndim(), 2);
        let (mut_a, mut_b) = unsafe {
            (
                mem::transmute::<&NdArray, &mut NdArray>(xs[0]),
                mem::transmute::<&NdArray, &mut NdArray>(xs[1]),
            )
        };
        let a = ndarray_ext::into_mat(mem::replace(mut_a, dummy_tensor()));
        let b = ndarray_ext::into_mat(mem::replace(mut_b, dummy_tensor()));
        // dot product
        let y = a.dot(&b).into_dyn();
        mem::replace(mut_a, a.into_dyn());
        mem::replace(mut_b, b.into_dyn());
        y
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let ga = ops::matmul(gy, &ops::swap_axes(inputs[1], 0, 1));
        let gb = ops::matmul(&ops::swap_axes(inputs[0], 0, 1), gy);
        vec![Some(ga), Some(gb)]
    }
}


#[inline(always)]
fn dummy_tensor() -> NdArray
{
    NdArray::default(ndarray::IxDyn(&[]))
}
