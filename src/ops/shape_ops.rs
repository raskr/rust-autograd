extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Shape;
pub struct Rank;
pub struct Size;


impl ops::Op for Shape {
    fn name(&self) -> &str
    {
        "Shape"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let shape = xs[0].shape().iter().map(|a| *a as f32).collect::<Vec<_>>();
        // safe unwrap
        NdArray::from_shape_vec(ndarray::IxDyn(&[shape.len()]), shape).unwrap()
    }
}

impl ops::Op for Rank {
    fn name(&self) -> &str
    {
        "Rank"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let x: &NdArray = xs[0];
        NdArray::from_elem(ndarray::IxDyn(&[1]), x.ndim() as f32)
    }
}

impl ops::Op for Size {
    fn name(&self) -> &str
    {
        "Size"
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }

    fn compute(&self, xs: &[&::NdArray], _: bool) -> ::NdArray
    {
        let x: &NdArray = xs[0];
        NdArray::from_elem(ndarray::IxDyn(&[1]), x.len() as f32)
    }
}
