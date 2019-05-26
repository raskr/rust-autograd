use op;
use tensor::Tensor;
use Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(&self, _: ::runtime::OpComputeContext<T>) -> op::ComputeResults<T> {
        vec![Err(::op::ComputeException::Delegate { to: 0 })]
    }

    fn grad(&self, _: &Tensor<T>, _: &[&Tensor<T>], _: &Tensor<T>) -> Vec<Option<Tensor<T>>> {
        vec![None]
    }
}
