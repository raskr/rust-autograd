extern crate ndarray;

use ndarray_ext;
use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct Zeros;
pub struct Ones;
pub struct Range;

pub struct ConvertToTensor {
    pub arr: NdArray,
}

impl ops::Op for Zeros {
    fn name(&self) -> &str
    {
        "Zeros"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let shape: &NdArray = xs[0];
        if let Some(a) = shape.as_slice() {
            ndarray_ext::zeros(a.iter().map(|&b| b as usize).collect::<Vec<_>>().as_slice())
        } else {
            ndarray_ext::zeros(
                shape
                    .iter()
                    .map(|&b| b as usize)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        }
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Ones {
    fn name(&self) -> &str
    {
        "Ones"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let shape: &NdArray = xs[0];
        if let Some(a) = shape.as_slice() {
            ndarray_ext::ones(a.iter().map(|&b| b as usize).collect::<Vec<_>>().as_slice())
        } else {
            ndarray_ext::ones(
                shape
                    .iter()
                    .map(|&b| b as usize)
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        }
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Range {
    fn name(&self) -> &str
    {
        "Range"
    }

    fn compute(&self, xs: &[&NdArray], _: bool) -> NdArray
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let x2 = xs[2];

        assert_eq!(x0.len(), 1);
        assert_eq!(x1.len(), 1);
        assert_eq!(x2.len(), 1);

        // safe unwrap
        let start = *x0.get(0).unwrap();
        let end = *x1.get(0).unwrap();
        let step = *x2.get(0).unwrap();

        ndarray::Array1::range(start, end, step).into_dyn()
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None, None, None]
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
