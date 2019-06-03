use crate::op;
use crate::tensor::Tensor;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(&self, _: crate::runtime::OpComputeContext<T>) -> op::ComputeResults<T> {
        vec![Err(crate::op::ComputeException::Delegate { to: 0 })]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}
