use crate::ndarray_ext;
use crate::ndarray_ext::NdArray;
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;

pub struct Zeros;
pub struct Ones;
pub struct Range;
pub struct ConvertToTensor<T: Float> {
    pub arr: NdArray<T>,
}
pub struct Scalar<T: Float> {
    pub val: T,
}

impl<T: Float> op::Op<T> for Scalar<T> {
    fn name(&self) -> &str {
        "Scalar"
    }

    fn compute<'v>(&self, _: crate::runtime::OpComputeContext<'v, T>) -> op::ComputeResults<'v, T> {
        vec![Ok(crate::ArrRepr::Owned(
            ndarray::arr0(self.val).into_dyn(),
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Zeros {
    fn name(&self) -> &str {
        "Zeros"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let shape = &xs[0];
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::zeros(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::zeros(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        vec![Ok(crate::ArrRepr::Owned(ret))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Ones {
    fn name(&self) -> &str {
        "Ones"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let shape = &xs[0];
        let ret = if let Some(a) = shape.as_slice() {
            ndarray_ext::ones(
                a.iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        } else {
            ndarray_ext::ones(
                shape
                    .iter()
                    .map(|&b| b.to_usize().unwrap())
                    .collect::<Vec<_>>()
                    .as_slice(),
            )
        };
        vec![Ok(crate::ArrRepr::Owned(ret))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}

impl<T: Float> op::Op<T> for Range {
    fn name(&self) -> &str {
        "Range"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        let xs = ctx.grab_inputs();
        let x0 = &xs[0];
        let x1 = &xs[1];
        let x2 = &xs[2];

        let true_shape = &[];
        if x0.shape() != true_shape || x1.shape() != true_shape || x2.shape() != true_shape {
            panic!("Inputs to `range` should be 0-ranked tensors");
        }

        let start = x0[ndarray::IxDyn(&[])];
        let end = x1[ndarray::IxDyn(&[])];
        let step = x2[ndarray::IxDyn(&[])];
        assert!(start < end, "`start` and `end` overlap.");
        vec![Ok(crate::ArrRepr::Owned(
            ndarray::Array1::range(start, end, step).into_dyn(),
        ))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None, None, None]
    }
}

impl<T: Float> op::Op<T> for ConvertToTensor<T> {
    fn name(&self) -> &str {
        "ConvertToTensor"
    }

    fn compute<'v>(&self, _: crate::runtime::OpComputeContext<'v, T>) -> op::ComputeResults<'v, T> {
        vec![Ok(crate::ArrRepr::Owned(self.arr.clone()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![]
    }
}
