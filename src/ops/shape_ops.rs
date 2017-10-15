extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Shape;
pub struct Rank;
pub struct Size;
pub struct Reshape;


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


impl ops::Op for Reshape {
    fn name(&self) -> &str
    {
        "Reshape"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let ret = xs[0].clone();
        let shape_arr: &NdArray = xs[1];
        let target = shape_arr
            .iter()
            .map(|&dim_size| if dim_size != -1. {
                dim_size as usize
            } else {
                let product: f32 = shape_arr.iter().product();
                ret.len() / -product as usize
            })
            .collect::<Vec<_>>();

        if let Ok(a) = ret.into_shape(ndarray::IxDyn(target.as_slice())) {
            a
        } else {
            panic!("reshape failed: shape incompatible")
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![
            Some(ops::apply_op(Reshape, &[gy, &ops::shape(inputs[0])])),
            None,
        ]
    }
}
