//! Defining things related to `ag::op::Op`.
//!
use crate::tensor::Tensor;
use crate::Float;

// Op can have multiple output arrays.
pub type ComputeResults<'v, T> = Vec<Result<crate::ArrRepr<'v, T>, ComputeException>>;

#[derive(Clone, Copy, Debug)]
/// This is an `exception`, not an error.
pub enum ComputeException {
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
/// type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
///
///     fn name(&self) -> &str
///     {
///         "Sigmoid"
///     }
///
///     // In this method, any errors caused by bad user-inputs should results in "panic".
///     // (`ag::op::ComputeException` represents an exception rather than an error.)
///     fn compute<'v>(
///         &self,
///         ctx: ag::runtime::OpComputeContext<'v, T>,
///     ) -> ag::op::ComputeResults<'v, T> {
///         let xs = ctx.grab_inputs();
///         let x = &xs[0];
///         // Use `ndarray::Array::mapv` for element-wise computation.
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(|a| ((a * half).tanh() * half) + half);
///         vec![Ok(ag::ArrRepr::Owned(y))]
///     }
///
///     fn grad(&self, gy: &ag::Tensor<T>, xs: &[&ag::Tensor<T>], y: &ag::Tensor<T>)
///         -> Vec<Option<ag::Tensor<T>>>
///     {
///         // Symbolic gradient of `x`
///         let gx = gy * (y - ag::square(y));
///         vec![Some(gx)]
///     }
/// }
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid<T: ag::Float>(x: &ag::Tensor<T>) -> ag::Tensor<T>
/// {
///     ag::Tensor::builder()
///         .set_inputs(vec![x])
///         .set_shape(x.shape())
///         .build(Sigmoid)
/// }
/// ```
pub trait Op<T: Float> {
    /// Name of this op
    fn name(&self) -> &str;

    /// Runs this op.
    fn compute<'v>(&self, ctx: crate::runtime::OpComputeContext<'v, T>) -> ComputeResults<'v, T>;

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
    fn grad(&self, gy: &Tensor<T>, xs: &[&Tensor<T>], y: &Tensor<T>) -> Vec<Option<Tensor<T>>>;
}
