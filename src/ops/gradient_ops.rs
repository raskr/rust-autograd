use op;
use tensor::Tensor;

pub struct StopGradient;

impl op::Op for StopGradient {
    fn name(&self) -> &str {
        "StopGradient"
    }

    fn compute(&self, _: ::runtime::OpComputeContext) -> op::ComputeResult {
        vec![Err(::op::ComputeError::Delegate { to: 0 })]
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>> {
        vec![None]
    }
}
