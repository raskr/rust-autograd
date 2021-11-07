use crate::ndarray_ext;
use crate::ndarray_ext::NdArray;
use crate::op;
use crate::Float;
use ndarray;

pub struct Zeros;
pub struct Ones;
pub struct ConvertToTensor<T: Float> {
    pub arr: NdArray<T>,
}
pub struct Scalar<T: Float> {
    pub val: T,
}

impl<T: Float> op::Op<T> for Scalar<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        ctx.append_output(ndarray::arr0(self.val).into_dyn());
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for Zeros {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
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
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for Ones {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
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
        ctx.append_output(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}

impl<T: Float> op::Op<T> for ConvertToTensor<T> {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        ctx.append_output(self.arr.clone());
        Ok(())
    }

    fn grad(&self, _: &mut crate::op::GradientContext<T>) {}
}
