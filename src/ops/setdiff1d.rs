extern crate ndarray;

use ndarray_ext::NdArray;
use ops;
use std::collections::hash_set::HashSet;
use std::iter::FromIterator;
use tensor::Tensor;


pub struct SetDiff1D;

impl ops::Op for SetDiff1D {
    fn name(&self) -> &str
    {
        "SetDiff1D"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0: &NdArray = xs[0];
        let x1: &NdArray = xs[1];

        let set_a: HashSet<isize> =
            HashSet::from_iter(x0.as_slice().unwrap().iter().map(|&a| a as isize));

        let set_b: HashSet<isize> =
            HashSet::from_iter(x1.as_slice().unwrap().iter().map(|&a| a as isize));

        let diff = set_a.difference(&set_b);

        let mut vec = diff.collect::<Vec<&isize>>();
        vec.sort();
        let vec = vec.into_iter().map(|&a| a as f32).collect::<Vec<f32>>();
        let len = vec.len();
        // safe unwrap
        NdArray::from_shape_vec(ndarray::IxDyn(&[len]), vec).unwrap()
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None]
    }
}
