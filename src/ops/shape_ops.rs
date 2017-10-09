extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Shape;
pub struct Rank;
pub struct Size;

pub struct ReshapeStatic {
    pub target_shape: Vec<Option<usize>>,
}

pub struct ReshapeGrad {
    pub target_shape: Vec<Option<usize>>,
}

pub struct ReshapeDynamic;


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

impl ops::Op for ReshapeStatic {
    fn name(&self) -> &str
    {
        "Reshape"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let ret = xs[0].clone();

        let target = self.target_shape
            .iter()
            .map(|opt| if let &Some(len) = opt {
                len
            } else {
                let a = self.target_shape.iter().fold(
                    1,
                    |acc, x| acc * x.unwrap_or(1),
                );
                ret.len() - if a == 1 { 0 } else { a }
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
        let op = ReshapeGrad { target_shape: self.target_shape.clone() };
        vec![Some(ops::apply_op(op, &[inputs[0], gy]))]
    }
}

impl ops::Op for ReshapeGrad {
    fn name(&self) -> &str
    {
        "ReshapeGrad"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x = xs[0];
        let gy: &NdArray = xs[1];
        let orig_shape = x.shape();
        // unwrap is safe
        gy.clone().into_shape(orig_shape).unwrap()
    }

    fn grad(&self, _: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        let gy = inputs[1];
        let op = ReshapeStatic { target_shape: self.target_shape.clone() };
        vec![None, Some(ops::apply_op(op, &[gy]))]
    }
}

impl ops::Op for ReshapeDynamic {
    fn name(&self) -> &str
    {
        "ReshapeDynamic"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let ret = xs[0].clone();
        let shape_arr: &NdArray = xs[1];

        let target_shape = match shape_arr.as_slice() {
            Some(slice) => slice.iter().map(|&a| a as usize).collect::<Vec<_>>(),
            None => {
                ndarray::Array::<usize, ndarray::IxDyn>::from_shape_fn(
                    shape_arr.shape(),
                    |i| shape_arr[i] as usize,
                ).as_slice()
                 .unwrap()  // safe unwrap
                 .to_vec()
            }
        };

        if let Ok(a) = ret.into_shape(ndarray::IxDyn(target_shape.as_slice())) {
            a
        } else {
            panic!("reshape failed: shape incompatible")
        }
    }

    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![
            Some(ops::apply_op(ReshapeDynamic, &[gy, &ops::shape(inputs[0])])),
            None,
        ]
    }
}
