extern crate ndarray;

use ndarray_ext::NdArray;
use tensor::Tensor;

pub type ComputeResult = Vec<Result<NdArray, ComputeException>>;

#[derive(Clone, Debug)]
/// This is an `exception`, not an error.
pub enum ComputeException {
    /// Computation finished correctly but delegates the result to its `to` th input.
    Delegate { to: usize },
    /// Computation finished correctly with no output
    NoOutput,
}

/// Operation trait. `Tensor` wraps trait-object of this.
///
/// # Implementing differentiable operations
///
/// Many of well-known ops are pre-defined in `ag::ops`, but you can also
/// implement custom ops by hand.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// type NdArray = ndarray::Array<f32, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl ag::op::Op for Sigmoid {
///
///     fn name(&self) -> &str
///     {
///         "Sigmoid"
///     }
///
///     // In this method, any errors caused by bad user-inputs should results in "panic".
///     // (`ag::op::ComputeException` represents an exception rather than an error.)
///     fn compute(&self, ctx: ag::runtime::OpComputeContext)
///         -> Vec<Result<NdArray, ag::op::ComputeException>>
///     {
///         let xs = ctx.grab_inputs();
///         let x = xs[0];
///         // Use `ndarray::Array::mapv` for element-wise computation.
///         let y = x.mapv(|a| ((a * 0.5).tanh() * 0.5) + 0.5);
///         vec![Ok(y)]
///     }
///
///     fn grad(&self, gy: &ag::Tensor, xs: &[&ag::Tensor], y: &ag::Tensor)
///         -> Vec<Option<ag::Tensor>>
///     {
///         // Symbolic gradient of `x`
///         let gx = gy * (y - ag::square(y));
///         vec![Some(gx)]
///     }
/// }
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid(x: &ag::Tensor) -> ag::Tensor
/// {
///     ag::Tensor::builder()
///         .set_inputs(vec![x])
///         .set_shape(x.shape())
///         .build(Sigmoid)
/// }
/// ```
pub trait Op {
    /// Name of this op
    fn name(&self) -> &str;

    /// Runs this op.
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
