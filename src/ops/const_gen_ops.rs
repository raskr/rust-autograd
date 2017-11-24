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
pub struct Scalar {
    pub val: f32,
}


impl ops::Op for Scalar {
    fn name(&self) -> &str
    {
        "Scalar"
    }

    fn compute(&self, _: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(ndarray::arr0(self.val).into_dyn())
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}

impl ops::Op for Zeros {
    fn name(&self) -> &str
    {
        "Zeros"
    }

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let shape: &NdArray = xs[0];
        if let Some(a) = shape.as_slice() {
            Ok(ndarray_ext::zeros(
                a.iter().map(|&b| b as usize).collect::<Vec<_>>().as_slice(),
            ))
        } else {
            Ok(ndarray_ext::zeros(
                shape
                    .iter()
                    .map(|&b| b as usize)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ))
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

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let shape: &NdArray = xs[0];
        if let Some(a) = shape.as_slice() {
            Ok(ndarray_ext::ones(
                a.iter().map(|&b| b as usize).collect::<Vec<_>>().as_slice(),
            ))
        } else {
            Ok(ndarray_ext::ones(
                shape
                    .iter()
                    .map(|&b| b as usize)
                    .collect::<Vec<_>>()
                    .as_slice(),
            ))
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

    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        let x0 = xs[0];
        let x1 = xs[1];
        let x2 = xs[2];

        let true_shape = &[];
        if x0.shape() != true_shape || x1.shape() != true_shape || x2.shape() != true_shape {
            return Err(::OpComputeErrorStatus::BadInput(
                "Inputs to `range` should be 0-ranked tensors".to_string(),
            ));
        }

        let start = x0[ndarray::IxDyn(&[])];
        let end = x1[ndarray::IxDyn(&[])];
        let step = x2[ndarray::IxDyn(&[])];

        if start > end {
            return Err(::OpComputeErrorStatus::BadInput(
                "Start and end of `range` is wrong.".to_string(),
            ));
        }

        Ok(ndarray::Array1::range(start, end, step).into_dyn())
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

    fn compute(&self, _: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        Ok(self.arr.clone())
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![]
    }
}
