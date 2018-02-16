extern crate ndarray;

use ndarray_ext::NdArray;
use tensor::Tensor;

pub type ComputeResult = Vec<Result<NdArray, ::errors::OpComputeErrorStatus>>;

/// Operation trait. `Tensor` wraps trait-object of this.
pub trait Op
{
    /// Name of this op
    fn name(&self) -> &str;

    /// Actually runs this op.
    fn compute(&self, ctx: ::runtime::OpComputeContext) -> ComputeResult;

    /// Returns symbolic gradients for input nodes by use of output gradient etc.
    ///
    /// # Arguments
    ///
    /// * `gy` - Symbolic representation of the gradient of `compute`'s return value
    /// * `xs` - Symbolic representation of `compute::xs`
    /// * `y` - Symbolic representation of `compute`'s return value
    ///
    /// NOTE:
    /// The number of return values must match `xs.len()`.
    fn grad(&self, gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>;
}
