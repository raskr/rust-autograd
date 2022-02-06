//! A collection of functions for manipulating `autograd::Tensor` objects
use ndarray;

use crate::ndarray_ext::{ArrayRng, NdArray};
use crate::tensor::{AsTensor, Tensor};
use crate::{Float, Graph};
use rand::Rng;

mod activation_ops;
mod array_ops;
pub(crate) mod basic_source_ops;
pub(crate) mod binary_ops;
mod blas_ffi;
pub(crate) mod const_gen_ops;
mod conv_ops;
pub(crate) mod dot_ops;
pub(crate) mod gradient_descent_ops;
mod gradient_ops;
mod graph_ops;
pub(crate) mod higher_order_ops;
pub(crate) mod hook_ops;
mod math_ops;
mod random_ops;
mod reduction_ops;
mod xent_ops;

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

impl<'graph, F: Float> Tensor<'graph, F> {
    /// Gets the `i` th float value of this tensor.
    ///
    /// Index `i` can be negative.
    ///
    /// ```
    /// use ndarray::{self, array};
    /// use autograd as ag;
    /// use ag::tensor_ops::*;
    ///
    /// ag::run(|g| {
    ///    let a = convert_to_tensor(array![[2., 3.], [4., 5.]], g);
    ///    let b = a.access_elem(2);
    ///    assert_eq!(b.eval(g).unwrap()[ndarray::IxDyn(&[])], 4.);
    /// });
    /// ```
    pub fn access_elem(self, i: isize) -> Tensor<'graph, F> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder(self.graph)
            .append_input(&self, false)
            .build(op)
    }
}

/// Symbolic gradient tensors of `xs` in the same order as `xs`'s
///
/// * `ys` - Targets of differentiation that are arbitrary shapes.
/// * `xs` - Tensors with which differentiate `ys`.
///
/// See the more useful helper: [crate::optimizers::grad_helper()]
///
/// ```
/// // Partial derivatives of `z = 2x^2 + 3y + 1`.
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops as T;
///
/// ag::run(|g| {
///     let x = g.placeholder("x", &[]);
///     let y = g.placeholder("y", &[]);
///     let z = 2.*x*x + 3.*y + 1.;
///
///     // dz/dy
///     let gy = T::grad(&[z], &[y])[0];
///     // dz/dx
///     let gx = T::grad(&[z], &[x])[0];
///
///     // ddz/dx (differentiates `z` again)
///     let ggx = T::grad(&[gx], &[x])[0];
///
///     // evaluation of symbolic gradients
///     assert_eq!(3., gy.eval(g).unwrap()[ndarray::IxDyn(&[])]);
///     assert_eq!(4., ggx.eval(g).unwrap()[ndarray::IxDyn(&[])]);
///
///     // dz/dx requires to fill the placeholder `x`
///     let gx = g.evaluator()
///         .push(gx)
///         .feed(x, ndarray::arr0(2.).view())
///         .run();
///     assert_eq!(8., gx[0].as_ref().unwrap()[ndarray::IxDyn(&[])]);
/// });
/// ```
pub fn grad<'graph, F: Float, A, B>(ys: &[A], xs: &[B]) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let g = ys[0].as_ref().graph();
    let ys: Vec<_> = ys.into_iter().map(|y| sum_all(y)).collect();
    let mut grads = crate::gradient::compute_gradients(ys.as_slice(), xs, None, g);
    let mut ret = Vec::with_capacity(xs.len());
    for x in xs {
        if let Some(gx) = grads.get(x) {
            ret.push(gx);
        } else {
            // not differentiable
            ret.push(zeros(&x.as_ref().shape(), g));
        }
    }
    ret
}

/// Computes `xs`'s gradients with `ys`'s already known gradients.
///
/// Almost same spec as `grad`'s except that you can pass `ys`s already known gradients.
/// If `ys_grads` are tensors filled with 1s, this function should be replaced with `grad`.
///
/// NOTE: Please be careful to match `ys_grads[i].shape` and `ys[i].shape`,
/// otherwise **undefined behavior** would happen.
///
/// # Arguments
/// * `ys` - Targets of differentiation.
/// * `xs` - tensors with which differentiate `ys`.
/// * `ys_grads` - Already known gradients of `ys`.
///
/// # Returns
/// Gradient tensors of `xs` in the same order as `xs`'graph.
pub unsafe fn grad_with_default<'graph, F: Float, A, B>(
    ys: &[A],
    xs: &[B],
    ys_grads: &[Tensor<'graph, F>],
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let g = ys[0].as_ref().graph();
    let mut grads = crate::gradient::compute_gradients(ys, xs, Some(ys_grads), g);
    let mut ret = Vec::with_capacity(xs.len());
    for x in xs {
        if let Some(gx) = grads.get(x) {
            ret.push(gx);
        } else {
            // not differentiable
            ret.push(zeros(&x.as_ref().shape(), g));
        }
    }
    ret
}

/// Computes jacobians for variables.
///
/// # Arguments
/// * `y` - Target of differentiation.
/// * `xs` - Tensors with which differentiate `ys`.
/// * `y_size` - (flattened) size of `y`
///
/// # Returns
/// Jacobians for each variable. Each one is a matrix of shape `(y_size, x size)`.
///
/// Note: the current implementation works correctly but is unoptimized for serious use.
///
/// ```
/// use autograd as ag;
/// use ag::prelude::*;
/// use ag::tensor_ops::*;
///
/// let mut env = ag::VariableEnvironment::new();
///
/// let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
/// let a = env.set(rng.standard_normal(&[4, 2]));
/// let b = env.set(rng.standard_normal(&[2, 3]));
///
/// env.run(|g| {
///    let a = g.variable(a);
///    let b = g.variable(b);
///    let c = matmul(a, b);
///    let j = jacobians(c, &[a, b], 4*3);
///
///    assert_eq!(j[0].eval(g).unwrap().shape(), &[4*3, 4*2]);
///    assert_eq!(j[1].eval(g).unwrap().shape(), &[4*3, 2*3]);
/// });
/// ```
pub fn jacobians<'graph, A, B, F: Float>(
    y_: A,
    xs_: &[B],
    objective_len: usize,
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let y = y_.as_ref();
    let mut vec_vec = Vec::with_capacity(objective_len);
    for i in 0..objective_len as isize {
        vec_vec.push(grad(&[y.access_elem(i)], xs_));
    }

    let len = xs_.len();
    let mut ret = Vec::with_capacity(len);
    // post process gradients
    for i in 0..len {
        // jac is matrix
        let mut jac = Vec::with_capacity(objective_len);
        for vec in &vec_vec {
            jac.push(expand_dims(flatten(&vec[i]), &[0]));
        }
        // (y size, x size)
        ret.push(concat(&jac, 0));
    }
    ret
}

/// (Experimental) Computes hessian vector product
pub fn _hessian_vector_product<'graph, A, B, C, F: Float>(
    ys: &[A],
    xs: &[B],
    vectors: &[C],
) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
    C: AsRef<Tensor<'graph, F>>,
{
    let grads = grad(ys, xs);
    let products = grads
        .into_iter()
        .zip(vectors)
        .map(|(g, v)| g.as_ref().clone() * v.as_ref().clone())
        .collect::<Vec<_>>();
    grad(products.as_slice(), xs)
}

/// Stops gradient propagation.
///
/// Guarantees that the gradient is not propagated to the tensors behind this
/// during gradient computation.
pub fn stop_gradient<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .set_differentiable(false)
        .build(gradient_ops::StopGradient)
}

/// Returns a `Tensor` representation of the input tensor's shape
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let x: ag::Tensor<f32> = zeros(&[2, 3], c);
///    let s = shape(x);
///    assert_eq!(&[2., 3.], s.eval(c).unwrap().as_slice().unwrap());
/// });
/// ```
pub fn shape<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    if let Some(id) = x.inner().shape {
        return g.tensor(id);
    }
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Shape)
}

/// Returns the (symbolic) size of the input tensor
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let a: ag::Tensor<f32> = zeros(&[4, 3], c);
///    let b = size(a);
///
///    assert_eq!(12., b.eval(c).unwrap()[ndarray::IxDyn(&[])]);
/// });
/// ```
pub fn size<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Size)
}

/// Returns the (symbolic) rank of the input tensor
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|c| {
///    let x: ag::Tensor<f32> = zeros(&[2, 3, 4], c);
///    let r = rank(x);
///    assert_eq!(3., r.eval(c).unwrap()[ndarray::IxDyn(&[])]);
/// });
/// ```
pub fn rank<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_differentiable(false)
        .build(array_ops::Rank)
}

/// Elementwise sine
pub fn sin<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sin)
}

/// Elementwise cosine
pub fn cos<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Cos)
}

/// Elementwise tangent
pub fn tan<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Tan)
}

/// Elementwise arcsin
pub fn asin<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Asin)
}

/// Elementwise arccos
pub fn acos<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Acos)
}

/// Elementwise arctan
pub fn atan<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Atan)
}

/// Elementwise hyperbolic sine
pub fn sinh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sinh)
}

/// Elementwise hyperbolic cosine
pub fn cosh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Cosh)
}

/// Elementwise hyperbolic tangent
pub fn tanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Tanh)
}

/// Elementwise hyperbolic arcsin
pub fn asinh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Asinh)
}

/// Elementwise hyperbolic arccos
pub fn acosh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Acosh)
}

/// Elementwise hyperbolic arctan
pub fn atanh<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Atanh)
}

/// Elementwise lgamma function
pub fn lgamma_f32<'graph, A>(x: A) -> Tensor<'graph, f32>
where
    A: AsRef<Tensor<'graph, f32>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Lgamma)
}

/// Elementwise lgamma function
pub fn lgamma_f64<'graph, A>(x: A) -> Tensor<'graph, f64>
where
    A: AsRef<Tensor<'graph, f64>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Lgamma)
}

/// Elementwise digamma function
///
/// NOTE: derivative not implemented
pub fn digamma_f32<'graph, A>(x: A) -> Tensor<'graph, f32>
where
    A: AsRef<Tensor<'graph, f32>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Digamma)
}

/// Elementwise digamma function
///
/// NOTE: derivative not implemented
pub fn digamma_f64<'graph, A>(x: A) -> Tensor<'graph, f64>
where
    A: AsRef<Tensor<'graph, f64>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .build(math_ops::Digamma)
}

#[doc(hidden)]
/// Gets n th tensor in `x`.
///
/// `x` must be a result of a multi-outputs op;
/// otherwise index-out-of-bounds error may happen.
pub fn nth_tensor<'graph, A, F: Float>(x: A, n: usize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .set_input_indices(&[n])
        .build(activation_ops::Identity)
}

/// Identity function without copy.
pub fn identity<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(activation_ops::Identity)
}

#[inline]
fn infer_bin_op_shape<'graph, A, B, F: Float>(
    g: &'graph Graph<F>,
    shape_a: A,
    shape_b: B,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    Tensor::builder(g)
        .append_input(shape_a.as_ref(), false)
        .append_input(shape_b.as_ref(), false)
        .build(array_ops::InferBinOpShape)
}

/// Elementwise addition.
///
/// This can be replaced with `+` operation of Tensor.
#[inline]
pub fn add<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::AddOp)
}

/// Element-wise subtraction.
///
/// This can be replaced with `-` operation of Tensor.
#[inline]
pub fn sub<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, &shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::SubOp)
}

/// Elementwise multiplication.
///
/// This can be replaced with `*` operation of Tensor.
#[inline]
pub fn mul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::MulOp)
}

/// Elementwise division.
///
/// This can be replaced with `/` operation of Tensor.
#[inline]
pub fn div<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = a.as_ref().graph();
    Tensor::builder(g)
        .set_shape(&infer_bin_op_shape(g, shape(a), shape(b)))
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(binary_ops::DivOp)
}

/// Elementwise sqrt
#[inline]
pub fn sqrt<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Sqrt)
}

/// Elementwise pow
pub fn pow<'graph, A, F: Float>(x: A, a: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Pow { a })
}

/// Elementwise base e (napier) logarithm
pub fn ln<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Ln)
}

/// Elementwise base 2 logarithm
pub fn log2<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Log2)
}

/// Elementwise base 10 logarithm
pub fn log10<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Log10)
}

/// Elementwise base e (napier) exponential
pub fn exp<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp)
}

/// Elementwise base 2 exponential
pub fn exp2<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp2)
}

/// Elementwise base 10 exponential
pub fn exp10<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .set_shape(&shape(x))
        .build(math_ops::Exp10)
}

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![1., 2., 3.], g);
///    let b = convert_to_tensor(array![3., 2., 1.], g);
///    let c = maximum(a, b);
///    assert_eq!(c.eval(g), Ok(array![3., 2., 3.].into_dyn()));
/// });
/// ```
pub fn maximum<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Maximum)
}

/// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![1., 2., 3.], g);
///    let b = convert_to_tensor(array![3., 2., 1.], g);
///    let c = minimum(a, b);
///    assert_eq!(c.eval(g), Ok(array![1., 2., 1.].into_dyn()));
/// });
/// ```
pub fn minimum<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Minimum)
}

/// Adds all input tensors, element-wise.
///
/// All the input tensors must have same shapes.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = ones((&[2, 2]), g);
///    let b = ones((&[2, 2]), g);
///    let c = ones((&[2, 2]), g);
///    let d = add_n(&[a, b, c]);
///
///    assert_eq!(d.eval(g).unwrap().shape(), &[2, 2]);
///    assert_eq!(d.eval(g), Ok(array![[3., 3.], [3., 3.]].into_dyn()));
/// });
/// ```
pub fn add_n<'graph, A, F: Float>(xs: &[A]) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let len = xs.len();
    assert_ne!(len, 0);
    if len == 1 {
        *xs[0].as_ref()
    } else {
        let g = xs[0].as_ref().graph();
        let mut b = Tensor::builder(g);
        for x in xs {
            b = b.append_input(x.as_ref(), false);
        }
        b.set_shape(&shape(xs[0])).build(array_ops::AddN)
    }
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// if `a[i] == b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![1., 2., 3.], g);
///    let b = convert_to_tensor(array![3., 2., 1.], g);
///    let c = equal(a, b);
///    assert_eq!(c.eval(g), Ok(ndarray::arr1(&[0., 1., 0.]).into_dyn()));
/// });
/// ```
pub fn equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Equal)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// if `a[i] != b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![1., 2., 3.], g);
///    let b = convert_to_tensor(array![3., 2., 1.], g);
///    let c = not_equal(a, b);
///    assert_eq!(c.eval(g), Ok(array![1., 0., 1.].into_dyn()));
/// });
/// ```
pub fn not_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::NotEqual)
}

/// Takes argmin along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[3., 4.], [6., 5.]], g);
///    let y = argmin(x, 1, false);
///
///    assert_eq!(y.eval(g), Ok(array![0., 1.].into_dyn()));
/// });
/// ```
pub fn argmin<'graph, A, F: Float>(x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ArgMin { axis, keep_dim };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Takes argmax along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[3., 4.], [6., 5.]], g);
///    let y = argmax(x, 1, false);
///
///    assert_eq!(y.eval(g), Ok(array![1., 0.].into_dyn()));
/// });
/// ```
pub fn argmax<'graph, A, F: Float>(x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ArgMax { axis, keep_dim };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Expands the shape (inserts axes).
///
/// Each axis can be negative.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3], g);
///    let b = expand_dims(a, &[0, 2]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[1, 3, 1]);
/// });
/// ```
pub fn expand_dims<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(array_ops::ExpandDims)
}

/// Remove the specified dims.
///
/// Each axis can be negative.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[1, 3, 1], g);
///    let b = squeeze(a, &[0, 2]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[3]);
/// })
/// ```
pub fn squeeze<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, false)
        .append_input(&axes.as_tensor(g), false)
        .build(array_ops::Squeeze)
}

/// Tiles the input tensor along specified axis.
///
/// Tiles input tensor `num` times along `axis`.
/// `axis` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 2.], [3., 3.]], g);
///    let y = tile(x, 0, 2);
///
///    assert_eq!(
///        y.eval(g),
///        Ok(array![[2., 2.], [3., 3.], [2., 2.], [3., 3.]].into_dyn())
///    );
/// });
/// ```
pub fn tile<'graph, A, F: Float>(x: A, axis: isize, num: usize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = array_ops::Tile { axis, num };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Limits all elements of `x` so as to be within `[min, max]`
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![2., 4., 6.], g);
///    let y = clip(x, 3., 5.);
///    assert_eq!(y.eval(g), Ok(ndarray::arr1(&[3., 4., 5.]).into_dyn()));
/// });
/// ```
pub fn clip<'graph, A, F: Float>(x: A, min: F, max: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = array_ops::Clip { min, max };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Takes max along specified axes.
///
/// Each of element of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = reduce_max(&x, &[0], false);
///    assert_eq!(y.eval(g), Ok(array![3., 4.].into_dyn()));
/// });
/// ```
pub fn reduce_max<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ReduceMax {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Takes min along specified axes.
///
/// Each of element of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = reduce_min(&x, &[0], false);
///    assert_eq!(y.eval(g), Ok(array![2., 1.].into_dyn()));
/// });
/// ```
pub fn reduce_min<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ReduceMin {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Sum up all the elements to a scalar value (0-D Tensor).
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = sum_all(&x);
///    assert_eq!(y.eval(g), Ok(ndarray::arr0(10.).into_dyn()));
/// });
/// ```
pub fn sum_all<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(reduction_ops::ReduceSumToScalar)
}

/// Average all the elements.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = mean_all(&x);
///    assert_eq!(y.eval(g), Ok(ndarray::arr0(2.5).into_dyn()));
/// });
/// ```
pub fn mean_all<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    sum_all(x) / size(x)
}

/// Takes sumation along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = reduce_sum(&x, &[1], false);
///    assert_eq!(y.eval(g), Ok(array![6., 4.].into_dyn()));
/// });
/// ```
pub fn reduce_sum<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ReduceSum {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Takes mean along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = reduce_mean(x, &[1], false);
///    assert_eq!(y.eval(g), Ok(array![3., 2.].into_dyn()));
/// });
/// ```
pub fn reduce_mean<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ReduceMean {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Takes product along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[2., 4.], [3., 1.]], g);
///    let y = reduce_prod(&x, &[1], false);
///    assert_eq!(y.eval(g), Ok(array![8., 3.].into_dyn()));
/// });
/// ```
pub fn reduce_prod<'graph, A, AT, F: Float>(x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = reduction_ops::ReduceProd {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Compute population variance along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x = convert_to_tensor(array![[1., 1.], [2., 2.]], g);
///    let y = reduce_variance(&x, &[1], false);
///    assert_eq!(y.eval(g), Ok(array![0., 0.].into_dyn()));
/// });
/// ```
pub fn reduce_variance<'graph, A, AT, F: Float>(
    x: A,
    axes: &AT,
    keep_dims: bool,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    reduce_mean(square(x - reduce_mean(x, axes, true)), axes, keep_dims)
}

/// Reshapes the input tensor without copy.
///
/// Only one element in `shape` can be `-1`.
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x: ag::Tensor<f32> = zeros(&[3, 2, 2], g);
///    let y = reshape(&x, &[3, -1]);
///    assert_eq!(y.eval(g), Ok(ag::ndarray_ext::zeros::<f32>(&[3, 4])));
/// });
/// ```
pub fn reshape<'graph, A, AT, F: Float>(x: A, shape: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&shape.as_tensor(g), false)
        .build(array_ops::Reshape)
}

/// Flattens the input tensor into 1-ranked (vector) without copy.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x: ag::Tensor<f32> = zeros(&[3, 2, 2], g);
///    let z = flatten(x);
///    assert_eq!(z.eval(g).unwrap().shape(), &[12]);
/// });
/// ```
pub fn flatten<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&scalar(F::one().neg(), g), false)
        .set_shape(&shape(x))
        .build(array_ops::Reshape)
}

/// Returns -1 if x < 0, 0 if x==0, 1 if x > 0, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![-5., 4.5, 0.], g);
///    let b = sign(a);
///    assert_eq!(
///        b.eval(g).unwrap().as_slice().unwrap(),
///        &[-1., 1., 0.]
///    );
/// });
/// ```
pub fn sign<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Sign)
}

/// Returns the largest integer less than or equal to a number, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![-0.2, 0., 0.2], g);
///    let b = abs(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(ndarray::arr1(&[0.2, 0., 0.2]).into_dyn())
///    );
/// });
/// ```
pub fn abs<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Abs)
}

/// Returns the largest integer less than or equal to a number, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], g);
///    let b = floor(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![-2., -2., -1.,  0.,  1.,  1.,  2.].into_dyn())
///    );
/// });
/// ```
pub fn floor<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Floor)
}

/// Performs the `-` operation.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![2., 3.], g);
///    let b = neg(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![-2., -3.].into_dyn())
///    );
/// });
/// ```
pub fn neg<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .build(math_ops::NegOp)
}

/// Takes square of the input.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![2., 3.], g);
///    let b = square(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![4., 9.].into_dyn())
///    );
/// });
/// ```
pub fn square<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Square)
}

/// Returns the `1/x`, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![2.], g);
///    let b = inv(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![0.5].into_dyn())
///    );
/// });
/// ```
pub fn inv<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(math_ops::Inv)
}

/// Returns the `1/sqrt(x)`, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![4.], g);
///    let b = inv_sqrt(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![0.5].into_dyn())
///    );
/// });
/// ```
pub fn inv_sqrt<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(math_ops::InvSqrt)
}

/// Returns the smallest integer greater than or equal to a number, element-wise.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0], g);
///    let b = ceil(a);
///    assert_eq!(
///        b.eval(g),
///        Ok(array![-1., -1., -0.,  1.,  2.,  2.,  2.].into_dyn())
///    );
///
/// });
/// ```
pub fn ceil<'graph, A, F: Float>(a: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .set_shape(&shape(a))
        .append_input(a.as_ref(), false)
        .build(math_ops::Ceil)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Greater)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::GreaterEqual)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::Lesser)
}

/// Compares a couple of tensors and returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser_equal<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(math_ops::LesserEqual)
}

/// Elementwise logistic sigmoid function.
pub fn sigmoid<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(activation_ops::Sigmoid)
}

/// Elementwise exponential linear unit.
///
/// See <https://arxiv.org/abs/1511.07289>
pub fn elu<'graph, A, F: Float>(x: A, alpha: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(activation_ops::ELU { alpha })
}

/// Elementwise rectified linear unit.
pub fn relu<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(activation_ops::ReLU)
}

/// Elementwise leaky relu.
///
/// In common, `alpha` is around 0.1 ~ 0.2.
///
/// https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf
pub fn leaky_relu<'graph, A, F: Float>(x: A, alpha: F) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    maximum(x, scalar(alpha, g) * x.as_ref())
}

/// Elementwise softplus.
pub fn softplus<'graph, A, F: Float>(x: A) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(activation_ops::Softplus)
}

/// Computes `log(sum(exp(x)))` along specified axis.
///
/// `axis` can be negative.
pub fn reduce_logsumexp<'graph, A, F: Float>(x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = math_ops::LogSumExp {
        axis,
        keep_dims: keep_dim,
    };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Log softmax function.
///
/// Computes `softmax(x)` along specified axis and
/// takes logarithm of it.
/// `axis` can be negative.
pub fn log_softmax<'graph, A, F: Float>(x: A, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .set_shape(&shape(x))
        .append_input(x.as_ref(), false)
        .build(xent_ops::LogSoftmax { axis })
}

/// Computes softmax along specified axis
///
/// `axis` can be negative.
pub fn softmax<'graph, A, F: Float>(x: A, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = activation_ops::Softmax { axis };
    Tensor::builder(g).append_input(x.as_ref(), false).build(op)
}

/// Computes `binary_cross_entropy(sigmoid(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(sigmoid)`.
///
/// # Arguments
/// * `y` - Tensor with arbitrary shape
/// * `t` - Ground-truth Tensor with same shape as `y`'graph
///
/// # Panics
/// When y.shape != t.shape.
///
/// # Returns
/// Loss tensor with same shape as inputs's shapes
pub fn sigmoid_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SigmoidCrossEntropy;
    Tensor::builder(g)
        .set_shape(&shape(y))
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// Computes `categorical_cross_entropy(softmax(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(softmax)`.
///
/// # Arguments
/// * `y` - Tensor with shape (batch_size, num_classes)
/// * `t` - Tensor with shape (batch_size, num_classes)
///
/// # Returns
/// Loss tensor with shape (batch_size, 1)
pub fn softmax_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SoftmaxCrossEntropy;
    Tensor::builder(g)
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// A variant of `softmax_cross_entropy`.
///
/// The behavior of this function is same as `softmax_cross_entropy`
/// except that `t` is **not** batch of one-hot distributions but batch of ground truth label ids.
///
/// # Arguments
/// * `y` - Tensor with shape (batch_size, num_classes)
/// * `t` - Tensor with shape (batch_size,) or (batch_size, 1)
///
/// # Returns
/// Loss tensor with shape (batch_size, 1)
pub fn sparse_softmax_cross_entropy<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let y = y.as_ref();
    let g = y.graph();
    let op = xent_ops::SparseSoftmaxCrossEntropy;
    Tensor::builder(g)
        .append_input(y.as_ref(), false)
        .append_input(t.as_ref(), false)
        .build(op)
}

/// Computes mean squared error
///
/// Note that the mean axis is the last one.
pub fn mean_squared_error<'graph, A, B, F: Float>(y: A, t: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    reduce_mean(square(y.as_ref() - t.as_ref()), &[-1], false)
}

/// Matrix multiplication.
///
/// Both `a` and `b` must be 2-ranked tensors.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 2], g);
///    let b: ag::Tensor<f32> = zeros(&[2, 3], g);
///    let c = matmul(a, b);
///    assert_eq!(c.eval(g).unwrap().shape(), &[4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
pub fn matmul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(dot_ops::MatMul {
            transpose_a: false,
            transpose_b: false,
        })
}

/// Computes tensor-dot-product (tensor contraction) along specified axes.
///
/// # Arguments
/// * `a` - First input tensor
/// * `b` - Second input tensor
/// * `a_axes` - `a`'s Contraction axes
/// * `b_axes` - `b`'s Contraction axes
///
/// NOTE:
///
/// * length of `a_axes` and `b_axes` must match.
/// * Each axis number can be negative.
/// * Supports only f32 and f64.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3, 4, 5], g);
///    let b: ag::Tensor<f32> = zeros(&[4, 3, 2], g);
///    let c = tensordot(a, b, &[1, 0], &[0, 1]);
///    assert_eq!(c.eval(g).unwrap().shape(), &[5, 2]);
/// });
/// ```
///
/// For detailed description,
/// see <https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html>.
pub fn tensordot<'graph, A, B, AT1, AT2, F: Float>(
    a: A,
    b: B,
    a_axes: &AT1,
    b_axes: &AT2,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
    AT1: AsTensor<'graph, F>,
    AT2: AsTensor<'graph, F>,
{
    let a = a.as_ref();
    let g = a.graph();
    // Preprocess
    let pre = &Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .append_input(&a_axes.as_tensor(g), false)
        .append_input(&b_axes.as_tensor(g), false)
        .build(dot_ops::TensordotPreprocess);
    let final_shape = nth_tensor(pre, 0);
    let perm_a = nth_tensor(pre, 1);
    let perm_b = nth_tensor(pre, 2);
    let new_shape_a = nth_tensor(pre, 3);
    let new_shape_b = nth_tensor(pre, 4);

    let a_reshaped = reshape(transpose(a, &perm_a), &new_shape_a);
    let b_reshaped = reshape(transpose(b, &perm_b), &new_shape_b);

    // matmul
    let mm = matmul(a_reshaped, b_reshaped);
    reshape(mm, &final_shape)
}

/// Batched matrix multiplication with inputs's transposition.
///
/// The rank of `a` and `b` must be equals.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[2, 3, 2, 4], g);
///    let b: ag::Tensor<f32> = zeros(&[2, 3, 2, 3], g);
///    let c = batch_matmul_t(a, b, true, false);
///    assert_eq!(c.eval(g).unwrap().shape(), &[2, 3, 4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
/// For detailed description, see <https://www.tensorflow.org/api_docs/python/tf/matmul>.
pub fn batch_matmul_t<'graph, A, B, F: Float>(
    a: A,
    b: B,
    trans_a: bool,
    trans_b: bool,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = dot_ops::BatchMatMul {
        transpose_a: trans_a,
        transpose_b: trans_b,
    };
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Batched matrix multiplication.
///
/// The rank of `a` and `b` must be equals.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = ones((&[2, 3, 4, 2]), g);
///    let b: ag::Tensor<f32> = ones((&[2, 3, 2, 3]), g);
///    let c = batch_matmul(a, b);
///    assert_eq!(c.eval(g).unwrap().shape(), &[2, 3, 4, 3]);
/// });
/// ```
///
/// This function supports only f32 and f64.
/// For detailed description, see <https://www.tensorflow.org/api_docs/python/tf/matmul>.
pub fn batch_matmul<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = dot_ops::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Takes diff between two tensors.
///
/// Returns the sorted, unique values in `a` that are not in `b`.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = convert_to_tensor(array![4., 1., 5., 2., 3., 6.], g);
///    let b = convert_to_tensor(array![[2., 3.], [1., 4.]], g);
///    let c = setdiff1d(a, b);
///    assert_eq!(
///        c.eval(g),
///        Ok(ndarray::arr1(&[5., 6.]).into_dyn())
///    )
/// });
/// ```
///
pub fn setdiff1d<'graph, A, B, F: Float>(a: A, b: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let a = a.as_ref();
    let g = a.graph();
    let op = array_ops::SetDiff1D;
    Tensor::builder(g)
        .append_input(a.as_ref(), false)
        .append_input(b.as_ref(), false)
        .build(op)
}

/// Permutes dimensions without copy.
///
/// It's like TensorFlow or NumPy's.
/// `x`'s rank (ndim) and `axes.len()` must match.
///
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[1, 2, 3, 4, 5], g);
///    let b = transpose(a, &[4, 2, 3, 0, 1]);
///    assert_eq!(b.eval(g).unwrap().shape(), &[5, 3, 4, 1, 2]);
/// });
/// ```
pub fn transpose<'graph, A, AT, F: Float>(x: A, axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = x.as_ref();
    let g = x.graph();
    let op = math_ops::Transpose { invert_axes: false };
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(&axes.as_tensor(g), false)
        .build(op)
}

/// Splits input tensors into parts.
///
/// Splits `x` into `sizes.len()` parts along `axis`.
///
/// The size of dimension of each part is `sizes[i]` on `axis`, but is
/// `x.shape[i]` on other axis (similar to TensorFlow's `split`).
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3, 7, 5], g);
///    let b = split(a, &[2, 3, 2], 1);
///
///    let evaluated = g.evaluator().extend(&[&b[0], &b[1], &b[2]]).run();
///    let e0 = &evaluated[0];
///    let e1 = &evaluated[1];
///    let e2 = &evaluated[2];
///
///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 2, 5]);
///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 3, 5]);
///    assert_eq!(e2.as_ref().unwrap().shape(), &[3, 2, 5]);
/// });
/// ```
pub fn split<'graph, A, F: Float>(x: A, sizes: &[usize], axis: isize) -> Vec<Tensor<'graph, F>>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    let len = sizes.len();
    let mut ret = Vec::with_capacity(len);
    for i in 0..len {
        let mut start_index = 0usize;
        for &size in sizes[..i].iter() {
            start_index += size;
        }
        let end_index = start_index + sizes[i];
        ret.push(
            Tensor::builder(g)
                .append_input(x.as_ref(), false)
                .build(array_ops::Split {
                    start_index: start_index as isize,
                    end_index: end_index as isize,
                    axis,
                }),
        );
    }
    ret
}

/// Slices the input tensor.
///
/// # Arguments
/// * `x` - Tensor with arbitrary shape.
/// * `starts` - Inclusive start indices for the dimensions.
/// * `ends` - End indices for the dimensions. **Each index is inclusive if it is negative and exclusive if it's not.**
///
/// NOTE: Negative values in `starts` and `ends` are counted from the back of the axis.
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 4], g);
///    let b = slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
///
///    assert_eq!(b.eval(g).unwrap().shape(), &[4, 2]);
/// });
/// ```
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 4], g);
///    let b = slice(a, &[0, 0], &[-2, 2]); // numpy equivalent is a[:-1, :2]
///
///    assert_eq!(b.eval(g).unwrap().shape(), &[3, 2]);
/// });
/// ```
pub fn slice<'graph, A, F: Float>(x: A, starts: &[isize], ends: &[isize]) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    // TODO: Make starts and ends ArrayLike
    assert_eq!(starts.len(), ends.len());
    let starts_ends = starts.iter().zip(ends.iter());

    let indices = starts_ends
        .map(|(s, &e)| {
            let e = if e == -1 {
                None
            } else {
                Some(if e < -1 { e + 1 } else { e })
            };
            let slice = ndarray::Slice::new(*s, e, 1);
            ndarray::SliceOrIndex::from(slice)
        })
        .collect::<Vec<ndarray::SliceOrIndex>>();

    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(array_ops::Slice { indices })
}

/// Concatenates input tensors along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[3, 2], g);
///    let b: ag::Tensor<f32> = zeros(&[3, 2], g);
///    let c: ag::Tensor<f32> = zeros(&[3, 2], g);
///    let d = concat(&[a, b, c], 0);
///
///    assert_eq!(d.eval(g).unwrap().shape(), &[9, 2]);
/// });
/// ```
pub fn concat<'graph, A, F: Float>(tensors: &[A], axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    assert_ne!(tensors.len(), 0);
    let g = tensors[0].as_ref().graph();
    let op = array_ops::Concat { axis };
    let mut b = Tensor::builder(g);
    for t in tensors {
        b = b.append_input(t.as_ref(), false);
    }
    b.build(op)
}

/// Gathers subviews from the input tensor.
///
/// Same spec as <https://www.tensorflow.org/api_docs/python/tf/gather>.
/// For example, this can be used for embedding vectors lookup etc.
///
/// Unlike `ag::gather`, `indices` can contain negative elements.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let param = zeros(&[5, 4, 8, 2], g);
///    let indices = convert_to_tensor(array![[5., -1., 3.], [2., 1., -2.]], g);
///    let y = gather_common(param, indices, 2);
///
///    assert_eq!(y.eval(g).unwrap().shape(), &[5, 4, 2, 3, 2])
/// });
/// ```
pub fn gather_common<'graph, A, B, F: Float>(param: A, indices: B, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let param = param.as_ref();
    let g = param.graph();
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: true,
    };
    Tensor::builder(g)
        .append_input(indices.as_ref(), false)
        .append_input(param.as_ref(), false)
        .build(op)
}

/// Gathers subviews from the input tensor.
///
/// Same spec as <https://www.tensorflow.org/api_docs/python/tf/gather>.
/// For example, this can be used for embedding vectors lookup etc.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let param = zeros(&[5, 4, 8, 2], g);
///    let indices = convert_to_tensor(array![[5., 4., 3.], [2., 1., 0.]], g);  // shape: (2, 3)
///    let y = gather(param, indices, 2);
///
///    assert_eq!(y.eval(g).unwrap().shape(), &[5, 4, 2, 3, 2])
/// });
/// ```
pub fn gather<'graph, A, B, F: Float>(param: A, indices: B, axis: isize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let param = param.as_ref();
    let g = param.graph();
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: false,
    };
    Tensor::builder(g)
        .append_input(indices.as_ref(), false)
        .append_input(param, false)
        .build(op)
}

/// Normalizes the input tensor with its mean and variance along specified axis.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let x: ag::Tensor<f32> = standard_normal(&[3, 4], g);
///    let y1 = normalize(x, &[0]);
///    let y2 = normalize(x, &[0]);
///
///    let evaluated = g.evaluator().extend(&[y1, y2]).run();
///    let e0 = &evaluated[0];
///    let e1 = &evaluated[1];
///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 4]);
///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 4]);
/// });
/// ```
pub fn normalize<'graph, A, AT, F: Float>(_x: A, _axes: &AT) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    AT: AsTensor<'graph, F>,
{
    let x = _x.as_ref();
    let g = x.graph();
    let axes = _axes.as_tensor(g);
    let mean = reduce_mean(x.as_ref(), &axes, true);
    let centered = x - mean;
    let variance = reduce_mean(square(centered), &axes, true);
    let em5 = scalar(F::from(1e-5).unwrap(), g);
    centered * inv_sqrt(variance + em5)
}

/// Applies batch normalization.
///
/// `scale` and `shift` should be shared variables.
/// Since normalization is performed along 1st axis of `x`,
/// both of them should have shape `(1, x.shape[1])`
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
/// use ag::prelude::*;
///
/// let mut env = ag::VariableEnvironment::new();
/// let scale = env.set(ag::ndarray_ext::ones::<f32>(&[1, 4]));
/// let shift = env.set(ag::ndarray_ext::zeros::<f32>(&[1, 4]));
///
/// env.run(|g| {
///    let x = standard_normal(&[3, 4], g);
///    let norm = batch_norm(x, g.variable(scale), g.variable(shift));
///
///    assert_eq!(norm.eval(g).unwrap().shape(), &[3, 4]);
/// });
/// ```
pub fn batch_norm<'graph, A, B, C, F: Float>(x: A, scale: B, shift: C) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
    C: AsRef<Tensor<'graph, F>> + Copy,
{
    normalize(x, &[0]) * scale.as_ref() + shift.as_ref()
}

use crate::graph::AsGraph;
use std::marker::PhantomData;
use num_traits::zero;

/// Converts an `ndarray::Array` to a `ag::Tensor`.
///
/// ```
/// use ndarray::array;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let arr = array![2., 3.];
///    let tensor = convert_to_tensor(arr.clone(), g);
///    assert_eq!(tensor.eval(g), Ok(arr.into_dyn()));
/// });
/// ```
pub fn convert_to_tensor<F: Float, D>(
    arr: ndarray::Array<F, D>,
    graph: &impl AsGraph<F>,
) -> Tensor<F>
where
    D: ndarray::Dimension,
{
    let arr = arr.into_dyn();
    let shape = Tensor::builder(graph).build(const_gen_ops::ConvertToTensor {
        arr: crate::ndarray_ext::shape_of(&arr),
    });
    Tensor::builder(graph)
        .set_shape(shape.as_ref())
        .build(const_gen_ops::ConvertToTensor { arr })
}

/// Generates a zero-ranked tensor from a scalar value.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = scalar(3., g);
///    println!("{}", a.eval(g).unwrap());  // => 3.
///    assert_eq!(a.eval(g).unwrap().shape().len(), 0);
/// });
/// ```
pub fn scalar<F: Float>(val: F, graph: &impl AsGraph<F>) -> Tensor<F> {
    let op = const_gen_ops::Scalar { val };
    Tensor::builder(graph)
        .set_shape(&convert_to_tensor(
            crate::ndarray_ext::scalar_shape(),
            graph,
        ))
        .build(op)
}

/// Outputs values sampled from the normal distribution.
pub fn random_normal<'graph, A, F: Float>(
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_normal_rng(Default::default(), shape, mean, stddev, graph)
}

/// Outputs values sampled from the normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_normal_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::RandomNormal::new(arr_rng, mean, stddev))
}

/// Outputs values sampled from the uniform distribution.
pub fn random_uniform<'graph, A, F: Float>(
    shape: &A,
    min: f64,
    max: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_uniform_rng(Default::default(), shape, min, max, graph)
}

/// Outputs values sampled from the uniform distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_uniform_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    min: f64,
    max: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::RandomUniform::new(arr_rng, min, max))
}

/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal<'graph, A, F: Float>(
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    standard_normal_rng(Default::default(), shape, graph)
}

/// Outputs values sampled from the standard normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn standard_normal_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let shape = shape;
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::StandardNormal::new(arr_rng))
}

/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform<'graph, A, F: Float>(
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    standard_uniform_rng(Default::default(), shape, graph)
}

/// Outputs values sampled from the standard uniform distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn standard_uniform_rng<'graph, F: Float, A, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::StandardUniform::new(arr_rng))
}

/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli<'graph, A, F: Float>(
    shape: &A,
    p: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    bernoulli_rng(Default::default(), shape, p, graph)
}

/// Outputs values sampled from the bernoulli distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn bernoulli_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    p: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::Bernoulli::new(arr_rng, p))
}

/// Outputs values sampled from the exponential distribution.
pub fn random_exp<'graph, A, F: Float>(
    shape: &A,
    lambda: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_exp_rng(Default::default(), shape, lambda, graph)
}

/// Outputs values sampled from the exponential distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_exp_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    lambda: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::Exponential::new(arr_rng, lambda))
}

/// Outputs values sampled from the gamma distribution.
pub fn random_gamma<'graph, A, F: Float>(
    shape: &A,
    shape_param: f64,
    scale: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    random_gamma_rng(Default::default(), shape, shape_param, scale, graph)
}

/// Outputs values sampled from the gamma distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn random_gamma_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    shape_param: f64,
    scale: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::Gamma::new(arr_rng, shape_param, scale))
}

/// Outputs values sampled from the log-normal distribution.
pub fn log_normal<'graph, A, F: Float>(
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    log_normal_rng(Default::default(), shape, mean, stddev, graph)
}

/// Outputs values sampled from the log-normal distribution.
///
/// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
pub fn log_normal_rng<'graph, A, F: Float, R: Rng + 'static>(
    arr_rng: ArrayRng<F, R>,
    shape: &A,
    mean: f64,
    stddev: f64,
    graph: &'graph impl AsGraph<F>,
) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    let t = shape.as_tensor(graph);
    Tensor::builder(graph)
        .append_input(&t, false)
        .set_shape(&t)
        .build(random_ops::LogNormal::new(arr_rng, mean, stddev))
}

/// Returns zeros with given shape.
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a: ag::Tensor<f32> = zeros(&[4, 2], g);
///    assert_eq!(a.eval(g), Ok(ndarray::Array2::<f32>::zeros((4, 2)).into_dyn()));
/// });
/// ```
pub fn zeros<'graph, A, F: Float>(shape: &A, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    Tensor::builder(graph)
        .append_input(&shape.as_tensor(graph), false)
        .build(const_gen_ops::Zeros)
}

/// Returns ones with given shape.
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::tensor_ops::*;
///
/// ag::run(|g| {
///    let a = ones((&[4, 2]), g);
///    assert_eq!(a.eval(g), Ok(ndarray::Array2::<f32>::ones((4, 2)).into_dyn()));
/// });
/// ```
pub fn ones<'graph, A, F: Float>(shape: &A, graph: &'graph impl AsGraph<F>) -> Tensor<'graph, F>
where
    A: AsTensor<'graph, F>,
{
    Tensor::builder(graph)
        .append_input(&shape.as_tensor(graph), false)
        .build(const_gen_ops::Ones)
}

/// 2D convolution.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
/// * `w`: Tensor with shape `(out_channel, channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - filter_h) / stride + 1`
///   * `out_w` = `(w + 2 * pad - filter_w) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn conv2d<'graph, A, B, F: Float>(x: A, w: B, pad: usize, stride: usize) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d::Conv2D {
            pad,
            stride,
            dilation: 1,
        })
}

/// 2D convolution with dilation.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
/// * `w`: Tensor with shape `(out_channel, in_channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
///   * `out_w` = `(w + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn dilated_conv2d<'graph, A, B, F: Float>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
    dilate: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d::Conv2D {
            pad,
            stride,
            dilation: dilate,
        })
}

/// 2D transposed convolution.
///
/// * `x`: Tensor with shape `(batch, in_channel, h, w)`
/// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `stride * (h - 1) - pad + filter_h`
///   * `out_w` = `stride * (w - 1) - pad + filter_w`
///
/// This function supports only f32 and f64.
pub fn conv2d_transpose<'graph, A, B, F: Float>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d_transpose::Conv2DTranspose {
            pad,
            stride,
            dilation: 1,
        })
}

/// 2D transposed convolution with dilation.
///
/// * `x`: Tensor with shape `(batch, in_channel, h, w)`
/// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
///
/// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `stride * (h - 1) - pad + (dilate * (filter_h - 1) + 1)`
///   * `out_w` = `stride * (w - 1) - pad + (dilate * (filter_w - 1) + 1)`
///
/// This function supports only f32 and f64.
pub fn dilated_conv2d_transpose<'graph, A, B, F: Float>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
    dilate: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .append_input(w.as_ref(), false)
        .build(conv_ops::conv2d_transpose::Conv2DTranspose {
            pad,
            stride,
            dilation: dilate,
        })
}

/// 2D max pooling.
///
/// * `x`: Tensor with shape `(batch, channel, h, w)`
///
/// Returns a tensor with shape `(batch, channel, out_h, out_w)`
///
/// where
///
///   * `out_h` = `(h + 2 * pad - pool_size) / stride + 1`
///   * `out_w` = `(w + 2 * pad - pool_size) / stride + 1`
///
/// This function supports only f32 and f64.
pub fn max_pool2d<'graph, A, F: Float>(
    x: A,
    pool_size: usize,
    pad: usize,
    stride: usize,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(conv_ops::max_pool2d::MaxPool2D {
            pad,
            stride,
            size: pool_size,
        })
}

/// Dropout
///
/// http://arxiv.org/abs/1207.0580
///
/// `XorShiftRng` is used internally.
/// If you need to specify a seed value or use any other `Rng`, use `dropout_rng` instead.
pub fn dropout<'graph, A, F: Float>(x: A, dropout_ratio: F, train: bool) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    dropout_rng(
        x,
        dropout_ratio,
        train,
        crate::ndarray_ext::get_default_rng(),
    )
}

/// Dropout
///
/// http://arxiv.org/abs/1207.0580
pub fn dropout_rng<'graph, A, F: Float, R: Rng + 'static>(
    x: A,
    dropout_ratio: F,
    train: bool,
    rng: R,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(random_ops::Dropout {
            train,
            arr_rng: ArrayRng::new(rng),
            dropout_ratio,
        })
}

/// Same as [crate::tensor::Tensor::map()]
pub fn map<'graph, A, F: Float>(
    x: A,
    f: fn(crate::ndarray_ext::NdArrayView<F>) -> NdArray<F>,
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x.as_ref(), false)
        .build(higher_order_ops::MapOp {
            phantom: PhantomData,
            f,
        })
}

/// Controls evaluation order of tensors
///
/// Same as [crate::Tensor::depends_on()].
pub fn control_dependencies<'graph, A, F: Float>(
    x: Tensor<'graph, F>,
    deps: &[A],
) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
{
    let g = x.graph();
    if let Some(x_input) = x.input_tensor(0, g) {
        let mut ctrl_deps = Tensor::builder(g).append_input(x_input, false);
        // requiring all deps
        for dep in deps {
            ctrl_deps = ctrl_deps.append_input(dep.as_ref(), false);
        }
        let new_x_input = ctrl_deps.build(graph_ops::ControlDependency);
        g.access_inner_mut(x.id).in_nodes[0].id = new_x_input.id;
        x
    } else {
        panic!("Source tensor cannot depend on any other tensors.");
    }
}

/// Assigns `y` to `x`, elementwise.
///
/// Internally uses ndarray::ArrayBase::assign as is.
/// Note that `x` must be a variable tensor.
pub fn assign<'graph, A, B, F: Float>(x: A, y: B) -> Tensor<'graph, F>
where
    A: AsRef<Tensor<'graph, F>> + Copy,
    B: AsRef<Tensor<'graph, F>> + Copy,
{
    let x = x.as_ref();
    let y = y.as_ref();
    let g = x.graph();
    Tensor::builder(g)
        .append_input(x, true)
        .append_input(y, false)
        .build(array_ops::Assign)
}

// method version
impl<'g, F: Float> Tensor<'g, F> {
    /// Same as [tensor_ops::reshape](reshape)
    #[inline]
    pub fn reshape<AT: AsTensor<'g, F>>(&self, shape: &AT) -> Tensor<'g, F> {
        reshape(self, shape)
    }
    /// Same as [tensor_ops::flatten](flatten)
    #[inline]
    pub fn flatten(&self) -> Tensor<'g, F> {
        flatten(self)
    }
    /// Same as [tensor_ops::squeeze](squeeze)
    #[inline]
    pub fn squeeze<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        squeeze(self, axes)
    }
    /// Same as [tensor_ops::expand_dims](expand_dims)
    #[inline]
    pub fn expand_dims<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        expand_dims(self, axes)
    }
    /// Same as [tensor_ops::transpose](transpose)
    #[inline]
    pub fn transpose<AT: AsTensor<'g, F>>(&self, axes: &AT) -> Tensor<'g, F> {
        transpose(self, axes)
    }

    /// Same as [tensor_ops::size](size)
    #[inline]
    pub fn size(&self) -> Tensor<'g, F> {
        size(self)
    }
    /// Same as [tensor_ops::rank](rank)
    #[inline]
    pub fn rank(&self) -> Tensor<'g, F> {
        rank(self)
    }
    /// Same as [tensor_ops::shape](shape)
    #[inline]
    pub fn shape(&self) -> Tensor<'g, F> {
        shape(self)
    }

    /// Same as [tensor_ops::reduce_sum](reduce_sum)
    #[inline]
    pub fn reduce_sum<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_sum(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_mean](reduce_mean)
    #[inline]
    pub fn reduce_mean<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_mean(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_prod](reduce_prod)
    #[inline]
    pub fn reduce_prod<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_prod(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_min](reduce_min)
    #[inline]
    pub fn reduce_min<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_min(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_max](reduce_max)
    #[inline]
    pub fn reduce_max<AT: AsTensor<'g, F>>(&self, axes: &AT, keep_dims: bool) -> Tensor<'g, F> {
        reduce_max(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::reduce_variance](reduce_variance)
    #[inline]
    pub fn reduce_variance<AT: AsTensor<'g, F>>(
        &self,
        axes: &AT,
        keep_dims: bool,
    ) -> Tensor<'g, F> {
        reduce_variance(self, axes, keep_dims)
    }
    /// Same as [tensor_ops::sum_all](sum_all)
    #[inline]
    pub fn sum_all<AT: AsTensor<'g, F>>(&self) -> Tensor<'g, F> {
        sum_all(self)
    }
    /// Same as [tensor_ops::mean_all](mean_all)
    #[inline]
    pub fn mean_all<AT: AsTensor<'g, F>>(&self) -> Tensor<'g, F> {
        mean_all(self)
    }
}
