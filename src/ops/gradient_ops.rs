use crate::op;
use crate::tensor::Tensor;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute<'v>(
        &self,
        ctx: crate::runtime::OpComputeContext<'v, T>,
    ) -> op::ComputeResults<'v, T> {
        vec![Ok(crate::ArrRepr::View(ctx.grab_inputs()[0].clone()))]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}
