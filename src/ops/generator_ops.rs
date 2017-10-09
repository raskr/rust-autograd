extern crate ndarray;

use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Zeros {
    pub shape: Vec<usize>,
}

pub struct Ones {
    pub shape: Vec<usize>,
}

pub struct Range {
    pub start: f32,
    pub end: f32,
    pub step: f32,
}

pub struct ConvertToTensor {
    pub arr: NdArray,
}

impl ops::Op for Zeros {
    fn name(&self) -> &str
    {
        "Zeros"
    }

    fn compute(&self, _: &[&NdArray], _: bool) -> NdArray
    {
        ndarray_ext::zeros(self.shape.as_slice())
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![]
    }
}

impl ops::Op for Ones {
    fn name(&self) -> &str
    {
        "Ones"
    }

    fn compute(&self, _: &[&NdArray], _: bool) -> NdArray
    {
        ndarray_ext::ones(self.shape.as_slice())
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![]
    }
}

impl ops::Op for Range {
    fn name(&self) -> &str
    {
        "Range"
    }

    fn compute(&self, _: &[&NdArray], _: bool) -> NdArray
    {
        ndarray::Array1::range(self.start, self.end, self.step).into_dyn()
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![]
    }
}

impl ops::Op for ConvertToTensor {
    fn name(&self) -> &str
    {
        "ConvertToTensor"
    }

    fn compute(&self, _: &[&NdArray], _: bool) -> NdArray
    {
        self.arr.clone()
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![]
    }
}
