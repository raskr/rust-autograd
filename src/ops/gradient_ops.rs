use ndarray_ext::NdArray;
use ops;
use tensor::Tensor;


pub struct StopGradients;

impl ops::Op for StopGradients {
    fn name(&self) -> &str
    {
        "StopGradients"
    }

    fn compute(&self, _: &[&NdArray]) -> Result<NdArray, ::OpComputeErrorStatus>
    {
        return Err(::OpComputeErrorStatus::Delegate { to: 0 });
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        vec![None]
    }
}
