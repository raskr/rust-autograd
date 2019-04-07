extern crate ndarray;

use ndarray_ext::{ArrRng, NdArray};
use rand::Rng;
use tensor::{ArrayLike, Tensor};
use Float;

mod activation_ops;
mod array_ops;
mod basic_source_ops;
pub mod binary_ops;
mod const_gen_ops;
mod conv_ops;
#[macro_use]
pub mod dot_ops;
pub mod gradient_descent_ops;
mod gradient_ops;
mod math_ops;
mod random_ops;
mod reduction_ops;
mod xent_ops;

impl<T: Float> Tensor<T> {
    /// Looks up a symbolic element from this tensor.
    ///
    /// Index `i` can be negative.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let ref a = ag::variable(ndarray::arr2(&[[2., 3.], [4., 5.]]));
    /// let ref b = a.get(2);
    ///
    /// assert_eq!(b.eval(&[]).unwrap()[ndarray::IxDyn(&[])], 4.);
    /// ```
    pub fn get(&self, i: isize) -> Tensor<T> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder().set_input(self).build(op)
    }
}

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

/// Returns gradient tensors wrt input tensors.
///
/// # Arguments
/// * `ys` - Targets of differentiation.
/// * `xs` - Tensors with which differentiate `ys`; so its length must be same as `ys`'s.
///
/// NOTE: Each of `ys` **must** be a scalar (0-ranked tensor); otherwise it causes **undefined**
/// behavior.
/// For multi dimensional objectives, do `reduce_sum`/`reduce_mean` for all dimensionality or
/// use [grad_with_default](fn.grad_with_default.html).
///
/// # Returns
/// Symbolic gradient tensors corresponding to `xs` in the same order as `xs`'s.
///
///
/// # Example
/// Partial derivatives of `z = 2x^2 + 3y + 1`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::placeholder::<f64>(&[]);
/// let ref y = ag::placeholder::<f64>(&[]);
/// let ref z = 2.*x*x + 3.*y + 1.;
///
/// // dz/dy
/// let ref gy = ag::grad(&[z], &[y])[0];
/// // dz/dx
/// let ref gx = ag::grad(&[z], &[x])[0];
///
/// // ddz/dx (differentiates `z` again)
/// let ref ggx = ag::grad(&[gx], &[x])[0];
///
/// // evaluation of symbolic gradients
/// assert_eq!(3., gy.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
/// assert_eq!(4., ggx.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
///
/// // dz/dx requires to fill the placeholder `x`
/// assert_eq!(8., gx.eval(&[(x, &ndarray::arr0(2.).into_dyn())]).unwrap()[ndarray::IxDyn(&[])]);
///
/// ```
pub fn grad<T: Float>(ys: &[&Tensor<T>], xs: &[&Tensor<T>]) -> Vec<Tensor<T>> {
    ::gradient::symbolic_gradients(ys, xs, &vec![None; ys.len()])
}

/// Returns gradient tensors wrt input tensors.
///
/// # Arguments
/// * `ys` - Targets of differentiation.
/// * `xs` - tensors with which differentiate `ys`.
/// * `output_grads` - Already known gradients of `ys`.
///
/// The length must be same as `ys`'s. If **each objective is not a scalar**,
/// you must pass the "Some" value. In most cases, it is initialized with 1s.
///
/// # Returns
/// Symbolic gradient tensors corresponding to `xs` in the same order as `xs`'s.
///
/// For detailed, see [grad](fn.grad.html).
pub fn grad_with_default<T: Float>(
    ys: &[&Tensor<T>],
    xs: &[&Tensor<T>],
    output_grads: &[&Tensor<T>],
) -> Vec<Tensor<T>> {
    ::gradient::symbolic_gradients(
        ys,
        xs,
        output_grads
            .into_iter()
            .map(|a| Some(a.as_ref()))
            .collect::<Vec<_>>()
            .as_slice(),
    )
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
/// ```
/// extern crate autograd as ag;
///
/// let ref a = ag::variable(ag::ndarray_ext::standard_normal::<f32>(&[4, 2]));
/// let ref b = ag::variable(ag::ndarray_ext::standard_normal::<f32>(&[2, 3]));
/// let ref c = ag::matmul(a, b);
/// let ref j = ag::jacobians(c, &[a, b], 4*3);
///
/// assert_eq!(j[0].eval(&[]).unwrap().shape(), &[4*3, 4*2]);
/// assert_eq!(j[1].eval(&[]).unwrap().shape(), &[4*3, 2*3]);
/// ```
pub fn jacobians<T: Float>(
    y: &Tensor<T>,
    xs: &[&Tensor<T>],
    objective_len: usize,
) -> Vec<Tensor<T>> {
    // TODO: remove map
    let vec_vec = (0..objective_len as isize)
        .map(|i| {
            // For each scalar objective, computes gradients for all variables
            ::gradient::symbolic_gradients(&[&y.get(i)], xs, &[None])
        })
        .collect::<Vec<Vec<_>>>();

    // post process gradients
    (0..xs.len())
        .map(|i| {
            // jac is matrix
            let jac = (0..objective_len)
                .map(|j| expand_dims(&flatten(&vec_vec[j][i]), &[0]))
                .collect::<Vec<_>>();
            // (y size, x size)
            concat(jac.iter().map(|a| a).collect::<Vec<_>>().as_slice(), 0)
        })
        .collect::<Vec<_>>()
}

/// (Experimental) Computes hessian vector product
///
/// `ys` must be scalars.
pub fn _hessian_vector_product<T: Float>(
    ys: &[&Tensor<T>],
    xs: &[&Tensor<T>],
    vectors: &[&Tensor<T>],
) -> Vec<Tensor<T>> {
    let grads =
        ::gradient::symbolic_gradients(ys, xs, &xs.iter().map(|_| None).collect::<Vec<_>>());

    let products = grads
        .iter()
        .zip(vectors)
        .map(|(g, &v)| g * v)
        .collect::<Vec<_>>();

    let products = products.iter().map(|a| a).collect::<Vec<_>>();

    ::gradient::symbolic_gradients(products.as_slice(), xs, &[None])
}

/// Stops gradient propagation.
///
/// Guarantees that the gradient is not propagated to the tensors behind this
/// during gradient computation.
pub fn stop_gradient<A: AsRef<Tensor<T>>, T: Float>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_differentiable(false)
        .build(gradient_ops::StopGradient)
}

/// Creates a shared variable tensor from an array object.
///
/// A shared variable can be mutated with in-place ops or gradient descent methods
/// implemented in `autograd::gradient_descent_ops`.
/// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f64> = ag::variable(ndarray::arr1(&[2.]));
/// let ref y: ag::Tensor<f64> = 3. * x;
///
/// assert_eq!(6., y.eval(&[]).unwrap()[0]);
/// ```
#[inline]
pub fn variable<T: Float, D: ndarray::Dimension>(arr: ndarray::Array<T, D>) -> Tensor<T> {
    let arr = arr.into_dyn();
    Tensor::builder()
        .set_shape(convert_to_tensor(::ndarray_ext::shape_of(&arr)))
        .set_variable_array(arr)
        .build(basic_source_ops::Variable)
}

/// Creates a placeholder tensor.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let x = ag::placeholder(&[2]);
///
/// // Fills placeholder, then eval
/// let arr = ndarray::arr1(&[1., 1.]).into_dyn();
/// assert_eq!(x.eval(&[(&x, &arr.clone())]), Some(arr));
/// ```
#[inline]
pub fn placeholder<T: Float>(shape_: &[isize]) -> Tensor<T> {
    let b = Tensor::builder().set_is_placeholder(true);
    let rank = shape_.len();
    let b = if rank == 0 || -1 != shape_[0] {
        b.set_shape(convert_to_tensor(
            NdArray::from_shape_vec(
                ndarray::IxDyn(&[rank]),
                shape_
                    .iter()
                    .map(|&x| T::from(x).unwrap())
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        ))
    } else {
        b
    };
    b.build(basic_source_ops::Placeholder)
}

/// Creates a constant tensor.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let arr = ndarray::arr1(&[0., 0., 0.]);
/// let ref con = ag::constant(arr.clone());
/// assert_eq!(con.eval(&[]), Some(arr.into_dyn()))
/// ```
#[inline]
pub fn constant<D, T>(arr: ndarray::Array<T, D>) -> Tensor<T>
where
    D: ndarray::Dimension,
    T: Float,
{
    let arr = arr.into_dyn();
    Tensor::builder()
        .set_shape(convert_to_tensor(::ndarray_ext::shape_of(&arr)))
        .set_constant_array(arr)
        .build(basic_source_ops::Const)
}

/// Returns the (symbolic) shape of input tensor
///
/// ```
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3]);
/// let ref s = ag::shape(x);
///
/// assert_eq!(&[2., 3.], s.eval(&[]).unwrap().as_slice().unwrap());
/// ```
pub fn shape<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    if let Some(ref inner) = x.as_ref().shape {
        inner.clone()
    } else {
        Tensor::builder()
            .set_input(x.as_ref())
            .set_differentiable(false)
            .build(array_ops::Shape)
    }
}

/// Returns the (symbolic) size of input tensor
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 3]);
/// let ref b = ag::size(a);
///
/// assert_eq!(12., b.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
/// ```
pub fn size<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_differentiable(false)
        .build(array_ops::Size)
}

/// Returns the (symbolic) rank of input tensor
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f32> = ag::zeros(&[2, 3, 4]);
/// let ref r = ag::rank(x);
///
/// assert_eq!(3., r.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
/// ```
pub fn rank<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_differentiable(false)
        .build(array_ops::Rank)
}

/// Elementwise sine
pub fn sin<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Sin)
}

/// Elementwise cosine
pub fn cos<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Cos)
}

/// Elementwise tangent
pub fn tan<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Tan)
}

/// Elementwise arcsin
pub fn asin<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Asin)
}

/// Elementwise arccos
pub fn acos<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Acos)
}

/// Elementwise arctan
pub fn atan<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Atan)
}

/// Elementwise hyperbolic sine
pub fn sinh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Sinh)
}

/// Elementwise hyperbolic cosine
pub fn cosh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Cosh)
}

/// Elementwise hyperbolic tangent
pub fn tanh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Tanh)
}

/// Elementwise hyperbolic arcsin
pub fn asinh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Asinh)
}

/// Elementwise hyperbolic arccos
pub fn acosh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Acosh)
}

/// Elementwise hyperbolic arctan
pub fn atanh<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Atanh)
}

#[doc(hidden)]
/// Gets n th tensor in `x`.
///
/// `x` must be a result of a multi-outputs op;
/// otherwise index-out-of-bounds error may happen.
pub fn nth_tensor<A, T>(x: A, n: usize) -> Tensor<T>
where
    T: Float,
    A: AsRef<Tensor<T>>,
{
    Tensor::builder()
        .set_input(x.as_ref())
        .set_input_indices(vec![n])
        .build(activation_ops::Identity)
}

/// Identity function without copy.
pub fn identity<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(activation_ops::Identity)
}

#[inline]
fn infer_bin_op_shape<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(
    shape_a: A,
    shape_b: B,
) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![shape_a.as_ref(), shape_b.as_ref()])
        .build(array_ops::InferBinOpShape)
}

#[inline]
fn bin_op_helper<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>, O: ::op::Op<T> + 'static>(
    a: A,
    b: B,
    op: O,
) -> Tensor<T> {
    let a_shape = a.as_ref().shape();
    let b_shape = b.as_ref().shape();
    Tensor::builder()
        .set_shape(infer_bin_op_shape(&a_shape, &b_shape))
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Addition.
///
/// `+` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[2]);
/// let ref b = ag::ones(&[2]);
/// let ref z: ag::Tensor<f32> = a + b;
/// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[2., 2.]).into_dyn()));
/// ```
pub fn add<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    bin_op_helper(a, b, binary_ops::AddOp)
}

/// Subtraction.
///
/// `-` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[2]);
/// let ref b = ag::ones(&[2]);
///
/// let ref z: ag::Tensor<f32> = a - b;
/// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[0., 0.]).into_dyn()));
/// ```
pub fn sub<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    bin_op_helper(a, b, binary_ops::SubOp)
}

/// Multiplication.
///
/// `*` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
///
/// let ref a = ag::ones(&[2]);
/// let ref b = ag::ones(&[2]);
/// let ref z: ag::Tensor<f32> = a * b;
/// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
/// ```
pub fn mul<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    bin_op_helper(a, b, binary_ops::MulOp)
}

/// Division.
///
/// `/` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[2]);
/// let ref b = ag::ones(&[2]);
/// let ref z: ag::Tensor<f32> = a / b;
/// assert_eq!(z.eval(&[]), Some(ndarray::arr1(&[1., 1.]).into_dyn()));
/// ```
pub fn div<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    bin_op_helper(a, b, binary_ops::DivOp)
}

/// Inplace multiplication.
/// This function takes `a`'s ownership.
/// Note that this **doesn't** support gradient propagation.
///
/// # Panics
///
/// In case of `a` is a `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
///
/// let a = ag::ones(&[2]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[2]);
/// let ref c = ag::mul_inplace(a, b);
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[0., 0.]).into_dyn()));
/// ```
pub fn mul_inplace<T: Float, A: AsRef<Tensor<T>>>(a: Tensor<T>, b: A) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![&a, b.as_ref()])
        .set_shape(a.shape())
        .build(binary_ops::InplaceMulOp)
}

/// Inplace division.
///
/// This function takes `a`'s ownership.
/// Note that this **doesn't** support gradient propagation.
///
/// # Panics
///
/// In case of `a` is a `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::ones(&[2]);
/// let ref c = ag::div_inplace(a, &ag::scalar(2.));
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[0.5, 0.5]).into_dyn()));
/// ```
pub fn div_inplace<T: Float, A: AsRef<Tensor<T>>>(a: Tensor<T>, b: A) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![&a, b.as_ref()])
        .set_shape(a.shape())
        .build(binary_ops::InplaceDivOp)
}

/// Inplace addition
///
/// Returns `a` after performing `a += b`.
/// This function takes `a`'s ownership.
///
/// # Panics
///
/// In case of `a` is a `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::ones(&[2]);
/// let ref b = ag::ones(&[2]);
/// let ref c = ag::add_inplace(a, b);
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[2., 2.]).into_dyn()));
/// ```
pub fn add_inplace<T: Float, A: AsRef<Tensor<T>>>(a: Tensor<T>, b: A) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![&a, b.as_ref()])
        .set_shape(a.shape())
        .build(binary_ops::InplaceAddOp)
}

/// Inplace subtraction
///
/// Returns `a` after performing `a -= b`.
/// This function takes `a`'s ownership.
///
/// # Panics
///
/// In case of `a` is a `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::sub_inplace(a, b);
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn()));
/// ```
pub fn sub_inplace<T: Float, A: AsRef<Tensor<T>>>(a: Tensor<T>, b: A) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![&a, b.as_ref()])
        .set_shape(a.shape())
        .build(binary_ops::InplaceSubOp)
}

/// Elementwise sqrt
pub fn sqrt<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Sqrt)
}

/// Elementwise pow
pub fn pow<T: Float, A: AsRef<Tensor<T>>>(x: A, a: T) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Pow { a })
}

/// Elementwise log
pub fn log<T: Float, A: AsRef<Tensor<T>>>(x: A, a: T) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Log { a })
}

/// Elementwise exponential
pub fn exp<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .set_shape(x.as_ref().shape())
        .build(math_ops::Exp)
}

/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
/// let ref c = ag::maximum(a, b);
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[3., 2., 3.]).into_dyn()));
/// ```
pub fn maximum<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::Maximum)
}

/// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
/// let ref c = ag::minimum(a, b);
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[1., 2., 1.]).into_dyn()));
/// ```
pub fn minimum<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::Minimum)
}

/// Adds all input tensors, element-wise.
///
/// All the input tensors must have same shapes.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::ones(&[2, 2]);
/// let ref d = ag::add_n(&[a, b, c]);
///
/// assert_eq!(d.eval(&[]).as_ref().unwrap().shape(), &[2, 2]);
/// assert_eq!(d.eval(&[]), Some(ndarray::arr2(&[[3., 3.], [3., 3.]]).into_dyn()));
/// ```
pub fn add_n<T: Float>(xs: &[&Tensor<T>]) -> Tensor<T> {
    let len = xs.len();
    assert_ne!(len, 0);
    if len == 1 {
        xs[0].clone()
    } else {
        Tensor::builder()
            .set_inputs(xs.to_vec())
            .set_shape(xs[0].shape())
            .build(array_ops::AddN)
    }
}

/// Compares two tensors and returns a binary tensor.
///
/// if `a[i] == b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
/// let ref c = ag::equal(a, b);
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[0., 1., 0.]).into_dyn()));
/// ```
pub fn equal<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::Equal)
}

/// Compares two tensors and returns a binary tensor.
///
/// if `a[i] != b[i]` then `return-value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]));
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]));
/// let ref c = ag::not_equal(a, b);
///
/// assert_eq!(c.eval(&[]), Some(ndarray::arr1(&[1., 0., 1.]).into_dyn()));
/// ```
pub fn not_equal<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::NotEqual)
}

/// Takes argmax along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[3., 4.], [6., 5.]]));
/// let ref y = ag::argmax(x, 1, false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[1., 0.]).into_dyn()));
/// ```
pub fn argmax<T: Float, A: AsRef<Tensor<T>>>(x: A, axis: isize, keep_dim: bool) -> Tensor<T> {
    let op = reduction_ops::ArgMax { axis, keep_dim };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Expands specified dims.
///
/// Each axis can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[3]);
/// let ref b = ag::expand_dims(a, &[0, 2]);
///
/// assert_eq!(b.eval(&[]).unwrap().shape(), &[1, 3, 1]);
/// ```
pub fn expand_dims<T: Float, A: AsRef<Tensor<T>>, AL: ArrayLike<T>>(x: A, axes: &AL) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(array_ops::ExpandDims)
}

/// Squeezes specified dims.
///
/// Each axis can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[1, 3, 1]);
/// let ref b = ag::squeeze(a, &[0, 2]);
///
/// assert_eq!(b.eval(&[]).unwrap().shape(), &[3]);
/// ```
pub fn squeeze<T: Float, A: AsRef<Tensor<T>>, AL: ArrayLike<T>>(x: A, axes: &AL) -> Tensor<T> {
    let op = array_ops::Squeeze;
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Tiles input tensor along specified axis.
///
/// Tiles input tensor `num` times along `axis`.
/// `axis` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 2.], [3., 3.]]));
/// let ref y = ag::tile(x, 0, 2);
///
/// assert_eq!(
///     y.eval(&[]),
///     Some(ndarray::arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn())
/// );
/// ```
pub fn tile<T: Float, A: AsRef<Tensor<T>>>(x: A, axis: isize, num: usize) -> Tensor<T> {
    let op = array_ops::Tile { axis, num };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Limits all elements of `x` so as to be within `[min, max]`
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr1(&[2., 4., 6.]));
/// let ref y = ag::clip(x, 3., 5.);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 4., 5.]).into_dyn()));
/// ```
pub fn clip<T: Float, A: AsRef<Tensor<T>>>(x: A, min: T, max: T) -> Tensor<T> {
    let op = array_ops::Clip { min, max };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Takes max along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_max(&x, &[0], false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 4.]).into_dyn()));
/// ```
pub fn reduce_max<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axes: &AL,
    keep_dims: bool,
) -> Tensor<T> {
    let op = reduction_ops::ReduceMax {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Takes min along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_min(&x, &[0], false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[2., 1.]).into_dyn()));
/// ```
pub fn reduce_min<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axes: &AL,
    keep_dims: bool,
) -> Tensor<T> {
    let op = reduction_ops::ReduceMin {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Takes sum along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_sum(&x, &[1], false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[6., 4.]).into_dyn()));
/// ```
pub fn reduce_sum<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axes: &AL,
    keep_dims: bool,
) -> Tensor<T> {
    let op = reduction_ops::ReduceSum {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Takes mean along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_mean(x, &[1], false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[3., 2.]).into_dyn()));
/// ```
pub fn reduce_mean<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axes: &AL,
    keep_dims: bool,
) -> Tensor<T> {
    let op = reduction_ops::ReduceMean {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Takes product along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_prod(&x, &[1], false);
///
/// assert_eq!(y.eval(&[]), Some(ndarray::arr1(&[8., 3.]).into_dyn()));
/// ```
pub fn reduce_prod<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axes: &AL,
    keep_dims: bool,
) -> Tensor<T> {
    let op = reduction_ops::ReduceProd {
        keep_dims,
        sparse_axes: false,
    };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &axes.as_tensor()])
        .build(op)
}

/// Reshapes input tensor.
///
/// Only one element in `shape` can be `-1`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f32> = ag::zeros(&[3, 2, 2]);
/// let ref y = ag::reshape(&x, &[3, -1]);
///
/// assert_eq!(y.eval(&[]), Some(ag::ndarray_ext::zeros::<f32>(&[3, 4])));
/// ```
pub fn reshape<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(x: A, shape: &AL) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &shape.as_tensor()])
        .build(array_ops::Reshape)
}

/// Flattens input tensor into 1-ranked (vector).
///
/// ```
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f32> = ag::zeros(&[3, 2, 2]);
/// let ref z = ag::flatten(x);
/// assert_eq!(z.eval(&[]).unwrap().shape(), &[12]);
/// ```
pub fn flatten<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &scalar(T::one().neg())])
        .set_shape(x.as_ref().shape())
        .build(array_ops::Reshape)
}

/// Returns -1 if x < 0, 0 if x==0, 1 if x > 0, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[-5., 4.5, 0.]));
/// let ref b = ag::sign(a);
/// assert_eq!(
///     b.eval(&[]).unwrap().as_slice().unwrap(),
///     &[-1., 1., 0.]
/// );
/// ```
pub fn sign<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::Sign)
}

/// Returns the largest integer less than or equal to a number, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[-0.2, 0., 0.2]));
/// let ref b = ag::abs(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[0.2, 0., 0.2]).into_dyn())
/// );
/// ```
pub fn abs<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::Abs)
}

/// Returns the largest integer less than or equal to a number, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]));
/// let ref b = ag::floor(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[-2., -2., -1.,  0.,  1.,  1.,  2.]).into_dyn())
/// );
/// ```
pub fn floor<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::Floor)
}

/// Performs the `-` operation.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[2., 3.]));
/// let ref b = ag::neg(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[-2., -3.]).into_dyn())
/// );
/// ```
pub fn neg<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::NegOp)
}

/// Takes square of the input.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[2., 3.]));
/// let ref b = ag::square(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[4., 9.]).into_dyn())
/// );
/// ```
pub fn square<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::Square)
}

/// Returns the 1/x, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[2.]));
/// let ref b = ag::reciprocal(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[0.5]).into_dyn())
/// );
/// ```
pub fn reciprocal<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(math_ops::Reciprocal)
}

/// Returns the smallest integer greater than or equal to a number, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]));
/// let ref b = ag::ceil(a);
/// assert_eq!(
///     b.eval(&[]),
///     Some(ndarray::arr1(&[-1., -1., -0.,  1.,  2.,  2.,  2.]).into_dyn())
/// );
/// ```
pub fn ceil<T: Float, A: AsRef<Tensor<T>>>(a: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(a.as_ref().shape())
        .set_input(a.as_ref())
        .build(math_ops::Ceil)
}

/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::Greater)
}

/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater_equal<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::GreaterEqual)
}

/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::Lesser)
}

/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser_equal<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(math_ops::LesserEqual)
}

/// Elementwise logistic sigmoid function.
pub fn sigmoid<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(activation_ops::Sigmoid)
}

/// Elementwise exponential linear unit.
///
/// See https://arxiv.org/abs/1511.07289
pub fn elu<T: Float, A: AsRef<Tensor<T>>>(x: A, alpha: T) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(activation_ops::ELU { alpha })
}

/// Elementwise rectified linear unit.
pub fn relu<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(activation_ops::ReLU)
}

/// Elementwise leaky relu.
///
/// In common, `alpha` is around 0.1 ~ 0.2.
///
/// See http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
pub fn leaky_relu<T: Float, A: AsRef<Tensor<T>>>(x: A, alpha: T) -> Tensor<T> {
    maximum(&x, scalar(alpha) * x.as_ref())
}

/// Elementwise softplus.
pub fn softplus<T: Float, A: AsRef<Tensor<T>>>(x: A) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(activation_ops::Softplus)
}

/// Computes `log(sum(exp(x)))` along specified axis.
///
/// `axis` can be negative.
pub fn reduce_logsumexp<T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    axis: isize,
    keep_dim: bool,
) -> Tensor<T> {
    let op = math_ops::LogSumExp {
        axis,
        keep_dims: keep_dim,
    };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Log softmax function.
///
/// Computes `softmax(x)` along specified axis and
/// takes logarithm of it.
/// `axis` can be negative.
pub fn log_softmax<T: Float, A: AsRef<Tensor<T>>>(x: A, axis: isize) -> Tensor<T> {
    Tensor::builder()
        .set_shape(x.as_ref().shape())
        .set_input(x.as_ref())
        .build(xent_ops::LogSoftmax { axis })
}

/// Computes softmax along specified axis
///
/// `axis` can be negative.
pub fn softmax<T: Float, A: AsRef<Tensor<T>>>(x: A, axis: isize) -> Tensor<T> {
    let op = activation_ops::Softmax { axis };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Computes `binary_cross_entropy(sigmoid(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(sigmoid)`.
///
/// # Arguments
/// * `y` - Tensor with arbitrary shape
/// * `t` - Ground-truth Tensor with same shape as `y`'s
///
/// # Panics
/// When y.shape != t.shape.
///
/// # Returns
/// Loss tensor with same shape as inputs's shapes
pub fn sigmoid_cross_entropy<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(
    y: A,
    t: B,
) -> Tensor<T> {
    let op = xent_ops::SigmoidCrossEntropy;
    Tensor::builder()
        .set_shape(y.as_ref().shape())
        .set_inputs(vec![y.as_ref(), t.as_ref()])
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
pub fn softmax_cross_entropy<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(
    y: A,
    t: B,
) -> Tensor<T> {
    let op = xent_ops::SoftmaxCrossEntropy;
    Tensor::builder()
        .set_inputs(vec![y.as_ref(), t.as_ref()])
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
pub fn sparse_softmax_cross_entropy<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(
    y: A,
    t: B,
) -> Tensor<T> {
    let op = xent_ops::SparseSoftmaxCrossEntropy;
    Tensor::builder()
        .set_inputs(vec![y.as_ref(), t.as_ref()])
        .build(op)
}

/// Matrix multiplication.
///
/// Both `a` and `b` must be 2-ranked tensors.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 2]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul(a, b);
///
/// assert_eq!(c.eval(&[]).unwrap().shape(), &[4, 3]);
/// ```
pub fn matmul<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    let op = dot_ops::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Matrix multiplication with inputs's transposition.
///
/// Similar specification as `matmul` but, if `transpose_a` is true, `a` is transposed
/// before actual matrix multiplication. It is the same for `transpose_b`.
///
/// The performance is better than explicitly computing like `ag::matmul(ag::transpose)`.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 4]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul_t(a, b, true, false);
///
/// assert_eq!(c.eval(&[]).unwrap().shape(), &[4, 3]);
/// ```
pub fn matmul_t<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(
    a: A,
    b: B,
    transpose_a: bool,
    transpose_b: bool,
) -> Tensor<T> {
    let op = dot_ops::MatMul {
        transpose_a,
        transpose_b,
    };
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Computes tensor-dot-product (tensor contraction) along specified axes.
///
/// # Arguments
/// * `a` - Input tensor
/// * `b` - Input tensor
/// * `a_axes` - Contraction axes
/// * `b_axes` - Contraction axes
///
/// Note1: length of a_axes and b_axes must match.
///
/// Note2: Each axis number can be negative.
///
/// ```
/// extern crate autograd as ag;
///
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 4, 5]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[4, 3, 2]);
/// let ref c = ag::tensordot(a, b, &[1, 0], &[0, 1]);
/// assert_eq!(c.eval(&[]).unwrap().shape(), &[5, 2]);
/// ```
///
/// For detailed description,
/// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html.
pub fn tensordot<A, B, AL, T>(a: A, b: B, a_axes: &AL, b_axes: &AL) -> Tensor<T>
where
    T: Float,
    A: AsRef<Tensor<T>>,
    B: AsRef<Tensor<T>>,
    AL: ArrayLike<T>,
{
    fn normalize_negative_axes<T: Float>(axes: &Tensor<T>, x_rank: &Tensor<T>) -> Tensor<T> {
        let ref zero = zeros(&axes.shape());
        let ge = greater_equal(axes, zero);
        let lt = lesser(axes, zero);
        add_inplace(mul_inplace(ge, axes), &mul_inplace(lt, &(axes + x_rank)))
    }

    fn preprocess<T: Float, AL: ArrayLike<T>>(
        x: &Tensor<T>,
        axes: &AL,
        flip: bool,
    ) -> (Tensor<T>, Tensor<T>) {
        let ref x_shape = x.shape();
        let ref x_rank = x.rank();
        let ref axes = normalize_negative_axes(&axes.as_tensor(), x_rank);
        let ref free = setdiff1d(&_range(&scalar(T::zero()), x_rank, &scalar(T::one())), axes);

        let free_dims = gather(x_shape, free, 0);
        let ref axes_dims = gather(x_shape, axes, 0);
        let ref prod_free_dims = reduce_prod(&free_dims, &[0], true);
        let ref prod_axes_dims = reduce_prod(axes_dims, &[0], true);

        let (perm, new_shape) = if flip {
            (
                concat(&[axes, free], 0),
                concat(&[prod_axes_dims, prod_free_dims], 0),
            )
        } else {
            (
                concat(&[free, axes], 0),
                concat(&[prod_free_dims, prod_axes_dims], 0),
            )
        };

        (reshape(&transpose(x, &perm), &new_shape), free_dims)
    }

    // main procedure
    let ((a_reshaped, a_free_dims), (b_reshaped, b_free_dims)) = (
        preprocess(a.as_ref(), a_axes, false),
        preprocess(b.as_ref(), b_axes, true),
    );
    let ref mm = matmul(&a_reshaped, &b_reshaped);
    let final_shape = concat(&[&a_free_dims, &b_free_dims], 0);
    reshape(mm, &final_shape)
}

/// Batched matrix multiplication.
///
/// The rank of `a` and `b` must be equals.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 3, 4, 2]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3, 2, 3]);
/// let ref c = ag::batch_matmul(a, b);
///
/// assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
/// ```
///
/// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
pub fn batch_matmul_t<T, A, B>(a: A, b: B, trans_a: bool, trans_b: bool) -> Tensor<T>
where
    T: Float,
    A: AsRef<Tensor<T>>,
    B: AsRef<Tensor<T>>,
{
    let op = dot_ops::BatchMatMul {
        transpose_a: trans_a,
        transpose_b: trans_b,
    };
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Batched matrix multiplication.
///
/// The rank of `a` and `b` must be equals.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[2, 3, 4, 2]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[2, 3, 2, 3]);
/// let ref c = ag::batch_matmul(a, b);
///
/// assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
/// ```
///
/// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
pub fn batch_matmul<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    let op = dot_ops::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Takes diff between two tensors.
///
/// Returns the sorted, unique values in `a` that are not in `b`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::constant(ndarray::arr1(&[4., 1., 5., 2., 3., 6.]));
/// let ref b = ag::constant(ndarray::arr2(&[[2., 3.], [1., 4.]]));
/// let ref c = ag::setdiff1d(a, b);
///
/// assert_eq!(
///     c.eval(&[]),
///     Some(ndarray::arr1(&[5., 6.]).into_dyn())
/// )
/// ```
///
pub fn setdiff1d<T: Float, A: AsRef<Tensor<T>>, B: AsRef<Tensor<T>>>(a: A, b: B) -> Tensor<T> {
    let op = array_ops::SetDiff1D;
    Tensor::builder()
        .set_inputs(vec![a.as_ref(), b.as_ref()])
        .build(op)
}

/// Permutes dimensions.
///
/// It's like TensorFlow or NumPy's.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[1, 2, 3, 4, 5]);
/// let ref b = ag::transpose(a, &[4, 2, 3, 0, 1]);
///
/// assert_eq!(b.eval(&[]).unwrap().shape(), &[5, 3, 4, 1, 2]);
/// ```
pub fn transpose<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(x: A, perm: &AL) -> Tensor<T> {
    let op = math_ops::Transpose { zip: true };
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), &perm.as_tensor()])
        .build(op)
}

/// Splits input tensors into parts.
///
/// Splits `x` into `sizes.len()` parts along `axis`.
///
/// The size of dimension of each part is `sizes[i]` on `axis`, but
/// `x.shape[i]` on other axis (similar to TensorFlow's `split`).
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 7, 5]);
/// let ref b = ag::split(a, &[2, 3, 2], 1);
///
/// let evaluated = ag::eval(&[&b[0], &b[1], &b[2]], &[]);
/// let e0 = &evaluated[0];
/// let e1 = &evaluated[1];
/// let e2 = &evaluated[2];
///
/// assert_eq!(e0.as_ref().unwrap().shape(), &[3, 2, 5]);
/// assert_eq!(e1.as_ref().unwrap().shape(), &[3, 3, 5]);
/// assert_eq!(e2.as_ref().unwrap().shape(), &[3, 2, 5]);
/// ```
pub fn split<T: Float, A: AsRef<Tensor<T>>>(x: A, sizes: &[usize], axis: isize) -> Vec<Tensor<T>> {
    (0..sizes.len())
        .map(|i| {
            let op = array_ops::Split {
                sizes: sizes.to_vec(),
                index: i,
                axis,
            };
            Tensor::builder().set_input(x.as_ref()).build(op)
        })
        .collect::<Vec<_>>()
}

/// Slices the input tensor.
///
/// # Arguments
/// * `x` - Tensor with arbitrary shape.
/// * `starts` - Start indices for each dimensions
/// * `ends` - End indices for each dimensions.
/// `-1` representing the last index is acceptable for each dimension.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[4, 4]);
/// let ref b = ag::slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
///
/// assert_eq!(b.eval(&[]).unwrap().shape(), &[4, 2]);
/// ```
pub fn slice<T: Float, A: AsRef<Tensor<T>>>(x: A, starts: &[isize], ends: &[isize]) -> Tensor<T> {
    // TODO: Make starts and ends ArrayLike
    assert_eq!(starts.len(), ends.len());
    let starts_ends = starts.iter().zip(ends.iter());

    let indices = starts_ends
        .map(|(s, e)| ndarray::Si(*s, if *e == -1 { None } else { Some(*e) }, 1))
        .collect::<Vec<ndarray::Si>>();

    let op = array_ops::Slice {
        indices: indices.into_boxed_slice(),
    };
    Tensor::builder().set_input(x.as_ref()).build(op)
}

/// Concatenates input tensors along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a: ag::Tensor<f32> = ag::zeros(&[3, 2]);
/// let ref b: ag::Tensor<f32> = ag::zeros(&[3, 2]);
/// let ref c: ag::Tensor<f32> = ag::zeros(&[3, 2]);
/// let ref d = ag::concat(&[a, b, c], 0);
///
/// assert_eq!(d.eval(&[]).unwrap().shape(), &[9, 2]);
/// ```
pub fn concat<T: Float>(tensors: &[&Tensor<T>], axis: isize) -> Tensor<T> {
    let op = array_ops::Concat { axis };
    Tensor::builder().set_inputs(tensors.to_vec()).build(op)
}

/// Gathers subviews from the input tensor.
///
/// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
/// For example, this can be used for embedding vectors lookup etc.
///
/// Unlike `ag::gather`, `indices` can contain negative elements.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref param = ag::constant(ag::ndarray_ext::zeros::<f32>(&[5, 4, 8, 2]));
/// let ref indices = ag::constant(ndarray::arr2(&[[5., -1., 3.], [2., 1., -2.]]));
/// let ref y = ag::gather_common(param, indices, 2);
///
/// assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])
/// ```
pub fn gather_common<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(
    param: A,
    indices: &AL,
    axis: isize,
) -> Tensor<T> {
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: true,
    };
    Tensor::builder()
        .set_inputs(vec![&indices.as_tensor(), param.as_ref()])
        .build(op)
}

/// Gathers subviews from the input tensor.
///
/// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
/// For example, this can be used for embedding vectors lookup etc.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref param = ag::constant(ag::ndarray_ext::zeros::<f32>(&[5, 4, 8, 2]));
/// let ref indices = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
/// let ref y = ag::gather(param, indices, 2);
///
/// assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])
/// ```
pub fn gather<AL, T, A>(param: A, indices: &AL, axis: isize) -> Tensor<T>
where
    T: Float,
    AL: ArrayLike<T>,
    A: AsRef<Tensor<T>>,
{
    let op = array_ops::Gather {
        axis,
        should_normalize_negative_indices: false,
    };
    Tensor::builder()
        .set_inputs(vec![&indices.as_tensor(), param.as_ref()])
        .build(op)
}

/// Normalizes the input tensor with its mean and variance along specified axis.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x: ag::Tensor<f32> = ag::standard_normal(&[3, 4]);
/// let ref y1 = ag::normalize(x, &[0]);
/// let ref y2 = ag::normalize(x, &[0]);
///
/// let evaluated = ag::eval(&[y1, y2], &[]);
/// let e0 = &evaluated[0];
/// let e1 = &evaluated[1];
/// assert_eq!(e0.as_ref().unwrap().shape(), &[3, 4]);
/// assert_eq!(e1.as_ref().unwrap().shape(), &[3, 4]);
/// ```
pub fn normalize<AL: ArrayLike<T>, T: Float, A: AsRef<Tensor<T>>>(x: A, axes: &AL) -> Tensor<T> {
    let x = x.as_ref();
    let axes = axes.as_tensor();
    let ref mean = reduce_mean(x, &axes, true);
    let ref centered = x - mean;
    let ref variance = reduce_mean(square(centered), &axes, true);
    let em5 = T::from(1e-5).unwrap();
    (x - mean) / sqrt(&(variance + em5))
}

/// Applies batch normalization.
///
/// `scale` and `shift` should be shared variables.
/// Since normalization is performed along 1st axis of `x`,
/// both of them should have shape `(1, x.shape[1])`
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::standard_normal(&[3, 4]);
/// let ref scale = ag::variable(ag::ndarray_ext::ones::<f32>(&[1, 4]));
/// let ref shift = ag::variable(ag::ndarray_ext::zeros::<f32>(&[1, 4]));
/// let ref norm = ag::batch_norm(x, scale, shift);
///
/// assert_eq!(norm.eval(&[]).unwrap().shape(), &[3, 4]);
/// ```
pub fn batch_norm<A, B, C, T>(x: A, scale: B, shift: C) -> Tensor<T>
where
    T: Float,
    A: AsRef<Tensor<T>>,
    B: AsRef<Tensor<T>>,
    C: AsRef<Tensor<T>>,
{
    normalize(x, &[0]) * scale.as_ref() + shift.as_ref()
}

/// Generates a zero-ranked tensor from a scalar value.
///
/// ```
/// extern crate autograd as ag;
///
/// let a = ag::scalar(3.);
/// println!("{}", a.eval(&[]).unwrap());  // => 3.
/// assert_eq!(a.eval(&[]).unwrap().shape(), &[]);
/// ```
pub fn scalar<T: Float>(val: T) -> Tensor<T> {
    let op = const_gen_ops::Scalar { val };
    Tensor::builder()
        .set_shape(convert_to_tensor(::ndarray_ext::scalar_shape()))
        .build(op)
}

/// Outputs values sampled from the normal distribution.
pub fn random_normal<T: Float, AL: ArrayLike<T>>(shape: &AL, mean: f64, stddev: f64) -> Tensor<T> {
    random_normal_rng(Default::default(), shape, mean, stddev)
}

/// Outputs values sampled from the normal distribution.
pub fn random_normal_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    mean: f64,
    stddev: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::RandomNormal::new(arr_rng, mean, stddev))
}

/// Outputs values sampled from the uniform distribution.
pub fn random_uniform<T: Float, AL: ArrayLike<T>>(shape: &AL, min: f64, max: f64) -> Tensor<T> {
    random_uniform_rng(Default::default(), shape, min, max)
}

/// Outputs values sampled from the uniform distribution.
pub fn random_uniform_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    min: f64,
    max: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::RandomUniform::new(arr_rng, min, max))
}

/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal<T: Float, AL: ArrayLike<T>>(shape: &AL) -> Tensor<T> {
    standard_normal_rng(Default::default(), shape)
}

/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::StandardNormal::new(arr_rng))
}

/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform<T: Float, AL: ArrayLike<T>>(shape: &AL) -> Tensor<T> {
    standard_uniform_rng(Default::default(), shape)
}

/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::StandardUniform::new(arr_rng))
}

/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli<T: Float, AL: ArrayLike<T>>(shape: &AL, p: f64) -> Tensor<T> {
    bernoulli_rng(Default::default(), shape, p)
}

/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    p: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::Bernoulli::new(arr_rng, p))
}

/// Outputs values sampled from the exponential distribution.
pub fn random_exp<T: Float, AL: ArrayLike<T>>(shape: &AL, lambda: f64) -> Tensor<T> {
    random_exp_rng(Default::default(), shape, lambda)
}

/// Outputs values sampled from the exponential distribution.
pub fn random_exp_rng<T: Float + 'static, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    lambda: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::Exponential::new(arr_rng, lambda))
}

/// Outputs values sampled from the gamma distribution.
pub fn random_gamma<T: Float, AL: ArrayLike<T>>(
    shape: &AL,
    shape_param: f64,
    scale: f64,
) -> Tensor<T> {
    random_gamma_rng(Default::default(), shape, shape_param, scale)
}

/// Outputs values sampled from the gamma distribution.
pub fn random_gamma_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    shape_param: f64,
    scale: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::Gamma::new(arr_rng, shape_param, scale))
}

/// Outputs values sampled from the log-normal distribution.
pub fn log_normal<T: Float, AL: ArrayLike<T>>(shape: &AL, mean: f64, stddev: f64) -> Tensor<T> {
    log_normal_rng(Default::default(), shape, mean, stddev)
}

/// Outputs values sampled from the log-normal distribution.
pub fn log_normal_rng<T: Float, AL: ArrayLike<T>, R: Rng + 'static>(
    arr_rng: ArrRng<T, R>,
    shape: &AL,
    mean: f64,
    stddev: f64,
) -> Tensor<T> {
    let shape = shape.as_tensor();
    Tensor::builder()
        .set_input(&shape)
        .set_shape(shape)
        .build(random_ops::LogNormal::new(arr_rng, mean, stddev))
}

/// Converts an `ndarray::Array` to a `ag::Tensor`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let arr = ndarray::arr1(&[2., 3.]);
/// let tensor = ag::convert_to_tensor(arr.clone());
/// assert_eq!(tensor.eval(&[]), Some(arr.into_dyn()));
/// ```
pub fn convert_to_tensor<T, D>(arr: ndarray::Array<T, D>) -> Tensor<T>
where
    T: Float,
    D: ndarray::Dimension,
{
    let arr = arr.into_dyn();
    let shape = {
        let op = const_gen_ops::ConvertToTensor {
            arr: ::ndarray_ext::shape_of(&arr),
        };
        Tensor::builder().build(op)
    };
    Tensor::builder()
        .set_shape(shape)
        .build(const_gen_ops::ConvertToTensor { arr })
}

/// Returns zeros with given shape.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a: ag::Tensor<f32> = ag::zeros(&[4, 2]);
/// assert_eq!(a.eval(&[]), Some(ndarray::Array2::<f32>::zeros((4, 2)).into_dyn()));
/// ```
pub fn zeros<T: Float, AL: ArrayLike<T>>(shape: &AL) -> Tensor<T> {
    Tensor::builder()
        .set_input(&shape.as_tensor())
        .build(const_gen_ops::Zeros)
}

/// Returns ones with given shape.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::ones(&[4, 2]);
/// assert_eq!(a.eval(&[]), Some(ndarray::Array2::<f32>::from_elem((4, 2), 1.).into_dyn()));
/// ```
pub fn ones<T: Float, AL: ArrayLike<T>>(shape: &AL) -> Tensor<T> {
    Tensor::builder()
        .set_input(&shape.as_tensor())
        .build(const_gen_ops::Ones)
}

/// Returns a range.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let start = 0.;
/// let end = 5.;
/// let step = 1.;
/// let ref z = ag::range(start, end, step);
///
/// assert_eq!(z.eval(&[]), Some(ndarray::Array1::range(0., 5., 1.).into_dyn()));
/// ```
pub fn range<T: Float>(start: T, end: T, step: T) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![&scalar(start), &scalar(end), &scalar(step)])
        .build(const_gen_ops::Range)
}

pub fn _range<T: Float, AL: ArrayLike<T>>(start: &AL, end: &AL, step: &AL) -> Tensor<T> {
    Tensor::builder()
        .set_inputs(vec![
            &start.as_tensor(),
            &end.as_tensor(),
            &step.as_tensor(),
        ])
        .build(const_gen_ops::Range)
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
pub fn conv2d<A, B>(x: A, w: B, pad: usize, stride: usize) -> Tensor<f32>
where
    A: AsRef<Tensor<f32>>,
    B: AsRef<Tensor<f32>>,
{
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), w.as_ref()])
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
pub fn dilated_conv2d<A, B, T>(x: A, w: B, pad: usize, stride: usize, dilate: usize) -> Tensor<f32>
where
    A: AsRef<Tensor<f32>>,
    B: AsRef<Tensor<f32>>,
{
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), w.as_ref()])
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
pub fn conv2d_transpose<A, B>(x: A, w: B, pad: usize, stride: usize) -> Tensor<f32>
where
    A: AsRef<Tensor<f32>>,
    B: AsRef<Tensor<f32>>,
{
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), w.as_ref()])
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
pub fn dilated_conv2d_transpose<A, B>(
    x: A,
    w: B,
    pad: usize,
    stride: usize,
    dilate: usize,
) -> Tensor<f32>
where
    A: AsRef<Tensor<f32>>,
    B: AsRef<Tensor<f32>>,
{
    Tensor::builder()
        .set_inputs(vec![x.as_ref(), w.as_ref()])
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
pub fn max_pool2d<T: Float, A: AsRef<Tensor<T>>>(
    x: A,
    pool_size: usize,
    pad: usize,
    stride: usize,
) -> Tensor<T> {
    Tensor::builder()
        .set_input(x.as_ref())
        .build(conv_ops::max_pool2d::MaxPool2D {
            pad,
            stride,
            size: pool_size,
        })
}
