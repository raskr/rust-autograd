use crate::ndarray_ext;
use crate::ndarray_ext::NdArray;
use crate::op;
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
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        ctx.append_output(Ok(ndarray::arr0(self.val).into_dyn()));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<T: Float> op::Op<T> for Zeros {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = &ctx.input(0);
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
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<T: Float> op::Op<T> for Ones {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let shape = &ctx.input(0);
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
        ctx.append_output(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}

impl<T: Float> op::Op<T> for Range {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let x0 = &ctx.input(0);
        let x1 = &ctx.input(1);
        let x2 = &ctx.input(2);

        let true_shape = &[];
        if x0.shape() != true_shape || x1.shape() != true_shape || x2.shape() != true_shape {
            panic!("Inputs to `range` should be 0-ranked tensors");
        }

        let start = x0[ndarray::IxDyn(&[])];
        let end = x1[ndarray::IxDyn(&[])];
        let step = x2[ndarray::IxDyn(&[])];
        assert!(start < end, "`start` and `end` overlap.");
        ctx.append_output(Ok(ndarray::Array1::range(start, end, step).into_dyn()));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None, None, None]);
    }
}

impl<T: Float> op::Op<T> for ConvertToTensor<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        ctx.append_output(Ok(self.arr.clone()));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![]);
    }
}
