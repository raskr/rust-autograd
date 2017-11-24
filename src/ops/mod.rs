extern crate ndarray;

use context::Context;
use ndarray_ext::NdArray;
use std::rc::Rc;
use tensor::{RawTensor, Tensor};
use tensor::ArrayLike;

mod array_ops;
mod gradient_ops;
mod random_ops;
mod activation_ops;
mod math_ops;
mod binary_ops;
mod xent_ops;
mod dot_ops;
mod reduction_ops;
mod const_gen_ops;


#[doc(hidden)]
/// Represents a operation node in a computation graph.
/// `Tensor` wraps trait-object of this.
pub trait Op {
    /// Name of this op
    fn name(&self) -> &str;

    /// Flag: inplace or not.
    fn inplace(&self) -> bool
    {
        false
    }

    /// Actually runs this op.
    ///
    /// N inputs, 1 output.
    #[allow(unused_variables)]
    fn compute(&self, xs: &[&NdArray]) -> Result<NdArray, OpComputeErrorStatus>
    {
        unimplemented!()
    }

    /// Actually runs this op.
    ///
    /// Inplace operators such as InplaceAddOp override this.
    #[allow(unused_variables)]
    fn compute_inplace(&self, xs: &mut [&mut NdArray]) -> Result<(), OpComputeErrorStatus>
    {
        unimplemented!()
    }

    /// Returns symbolic gradients for input nodes by use of output gradient etc.
    ///
    /// # Arguments
    /// * `gy` - Symbolic representation of the gradient of `compute`'s return value
    /// * `xs` - Symbolic representation of `compute::xs`
    /// * `y` - Symbolic representation of `compute`'s return value
    ///
    /// NOTE:
    /// The number of return values must match `inputs.len()`.
    fn grad(&self, gy: &Tensor, xs: &[&Tensor], y: &Tensor) -> Vec<Option<Tensor>>;
}

/// Error statuses in Op#compute.
#[derive(Clone, Debug)]
pub enum OpComputeErrorStatus {
    /// This denotes that the op didn't any computation with no errors; i.e., this op
    /// delegates the result to its input node at the index `to`.
    Delegate { to: usize },
    /// Could'nt compute output array because of bad inputs.
    BadInput(String),
}

impl Tensor {
    /// Gets a symbolic element from this tensor with shape `[]`.
    ///
    /// Index `i` can be negative.
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut ctx = ag::Context::new();
    /// let ref a = ag::variable(ndarray::arr2(&[[2., 3.], [4., 5.]]), &mut ctx);
    /// let ref b = a.get(2);
    ///
    /// assert_eq!(b.eval(&mut ctx)[ndarray::IxDyn(&[])], 4.);
    /// ```
    pub fn get(&self, i: isize) -> Tensor
    {
        apply_op(
            array_ops::IndexOp { index: i },
            &[self],
            Some(convert_to_tensor(::ndarray_ext::scalar_shape())),
        )
    }
}


#[doc(hidden)]
#[inline(always)]
/// Helper. Generates a symbolic tensor.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[4, 2]);
/// let ref v = ag::zeros(&[2, 3]);
/// let ref b = ag::zeros(&[4, 3]);
/// let ref z = ag::matmul(a, v) + b;
/// let mut vars = [a, v, b, z];
/// // `sort_by_key` don't reverse the order of `a` and `v`
/// vars.sort_by_key(|a| a.top_rank);
/// assert!(vars == [a, v, b, z])
/// ```
pub fn apply_op<T: Op + 'static>(op: T, inputs: &[&Tensor], shape: Option<Tensor>) -> Tensor
{
    Tensor(Rc::new(RawTensor {
        op: Box::new(op),
        inputs: inputs.iter().map(|a| (*a).clone()).collect::<Vec<Tensor>>(),
        top_rank: inputs
            .iter()
            .map(|a| a.top_rank)
            .max()
            .map(|a| a + 1)
            .unwrap_or(0),
        shape,
    }))
}

pub struct DummyOp {
    pub name: String,
}

impl Op for DummyOp {
    fn name(&self) -> &str
    {
        &self.name
    }

    fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
    {
        unreachable!(
            "must not be called ({}#grad). This is probably bug.",
            self.name
        )
    }

    fn compute(&self, _: &[&::NdArray]) -> Result<NdArray, OpComputeErrorStatus>
    {
        let msg = if self.name == "PH" {
            "Wrong `Context` object usage,\
            or there exists placeholder(s) couldn't get initial value"
        } else if self.name == "Variable" {
            "Current graph evaluation context doesn't match with what generated this variable."
        } else if self.name == "Constant" {
            "Current graph evaluation context doesn't match with what generated this constant."
        } else {
            unreachable!()
        };
        panic!(msg);
    }
}

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------


/// Returns gradient tensors wrt input tensors.
///
/// # Arguments
/// * `ys` - Targets of differentiation.
/// * `xs` - tensors with which differentiate `ys`.
/// So the length must be same as `ys`'s.
///
/// NOTE: Each objective must be a scalar (0-ranked tensor).
/// For multi dimensional objectives, use [grad_with_default](ops.fn.grad_with_default.html).
///
/// # Returns
/// Symbolic gradient tensors corresponding to `xs` in the same order as `xs`
///
///
/// # Example
/// Partial derivatives of `z = 2x^2 + 3y + 1`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::placeholder(&[]);
/// let ref y = ag::placeholder(&[]);
/// let ref z = 2*x*x + 3*y + 1;
///
/// // dz/dy
/// let ref g1 = ag::grad(&[z], &[y])[0];
/// // dz/dx
/// let ref g2 = ag::grad(&[z], &[x])[0];
///
/// // ddz/dx (differentiates `z` again)
/// let ref gg = ag::grad(&[g2], &[x])[0];
///
/// // evaluation of symbolic gradients
/// let mut ctx = ag::Context::new();
/// assert_eq!(3., g1.eval(&mut ctx)[ndarray::IxDyn(&[])]);
/// assert_eq!(4., gg.eval(&mut ctx)[ndarray::IxDyn(&[])]);
///
/// // dz/dx requires to fill the placeholder `x`
/// ctx.feed_input(x, ndarray::arr0(2.));
/// assert_eq!(8., g2.eval(&mut ctx)[ndarray::IxDyn(&[])]);
///
/// ```
pub fn grad(ys: &[&Tensor], xs: &[&Tensor]) -> Vec<Tensor>
{
    ::gradient::symbolic_gradients(ys, xs, &ys.iter().map(|_| None).collect::<Vec<_>>())
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
/// Symbolic gradient tensors corresponding to `xs` in the same order as `xs`
///
/// For defailed, see [grad](ops.fn.grad.html).
pub fn grad_with_default(ys: &[&Tensor], xs: &[&Tensor], output_grads: &[&Tensor]) -> Vec<Tensor>
{
    ::gradient::symbolic_gradients(
        ys,
        xs,
        output_grads
            .into_iter()
            .map(|&a| Some(a))
            .collect::<Vec<_>>()
            .as_slice(),
    )
}


/// Computes jacobians for variables.
///
/// # Arguments
/// * `objective` - Target of differentiation.
/// * `variables` - Variable tensors with which differentiate `objective`.
/// * `objective_len` - (flattened) Length of `objective`
///
/// # Returns
/// Jacobians for each variable. Each one is matrix of shape `(objective_len, variable size)`.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::variable(ag::ndarray_ext::standard_normal(&[4, 2]), &mut ctx);
/// let ref b = ag::variable(ag::ndarray_ext::standard_normal(&[2, 3]), &mut ctx);
/// let ref c = ag::matmul(a, b);
/// let ref j = ag::jacobians(c, &[a, b], 4*3);
///
/// assert_eq!(j[0].eval(&mut ctx).shape(), &[4*3, 4*2]);
/// assert_eq!(j[1].eval(&mut ctx).shape(), &[4*3, 2*3]);
/// ```
pub fn jacobians(objective: &Tensor, variables: &[&Tensor], objective_len: usize) -> Vec<Tensor>
{
    // TODO: remove map
    let vec_vec = (0..objective_len as isize)
        .map(|i| {
            // For each scalar objective, computes gradients for all variables
            ::gradient::symbolic_gradients(&[&objective.get(i)], variables, &[None])
        })
        .collect::<Vec<Vec<_>>>();

    // post process gradients
    (0..variables.len())
        .map(|i| {
            // jac is matrix
            let jac = (0..objective_len)
                .map(|j| expand_dims(&flatten(&vec_vec[j][i]), &[0]))
                .collect::<Vec<_>>();
            // (objective_len, variable size)
            concat(jac.iter().map(|a| a).collect::<Vec<_>>().as_slice(), 0)
        })
        .collect::<Vec<_>>()
}


/// (Experimental) Computes hessian vector product
///
/// `ys` must be scalars.
pub fn _hessian_vector_product(ys: &[&Tensor], xs: &[&Tensor], vectors: &[&Tensor]) -> Vec<Tensor>
{
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


/// Stops gradients
///
/// Make sure that the gradient is not propagated to the tensors behind this.
pub fn stop_gradients(x: &Tensor) -> Tensor
{
    apply_op(gradient_ops::StopGradients, &[x], Some(x.shape()))
}


/// Creates a shared variable tensor from rust-ndarray's array object.
///
/// The shared variable behaves like any other tensors, except that
/// it can be optimized with gradient descent methods
/// implemented in `autograd::sgd`.
/// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x: ag::Tensor = ag::variable(ndarray::arr1(&[2.]), &mut ctx);
/// let ref y: ag::Tensor = 3 * x;
///
/// assert_eq!(6., y.eval(&mut ctx)[0]);
/// assert!(ctx.variables.contains_key(x));
/// assert_eq!(ctx.variables.get(x).unwrap(), &ndarray::arr1(&[2.]).into_dyn());
/// ```
#[inline]
pub fn variable<T>(arr: ndarray::Array<f32, T>, ctx: &mut Context) -> Tensor
where
    T: ndarray::Dimension,
{
    let arr = arr.into_dyn();
    let tensor = Tensor(Rc::new(RawTensor {
        op: Box::new(DummyOp { name: "Variable".to_string() }),
        inputs: vec![],
        top_rank: 0,
        shape: Some(convert_to_tensor(::ndarray_ext::shape_of(&arr))),
    }));
    ctx.variables.insert(tensor.clone(), arr);
    tensor
}


/// Creates a placeholder tensor.
///
/// See [Context](struct.Context.html).
#[inline]
pub fn placeholder(shape_: &[isize]) -> Tensor
{
    let rank = shape_.len();
    let shape = if rank != 0 && -1 == shape_[0] {
        // dynamic placeholder
        None
    } else {
        let arr = NdArray::from_shape_vec(
            ndarray::IxDyn(&[rank]),
            shape_.iter().map(|&x| x as f32).collect::<Vec<_>>(),
        ).unwrap();
        Some(convert_to_tensor(arr))
    };

    Tensor(Rc::new(RawTensor {
        op: Box::new(DummyOp { name: "PH".to_string() }),
        inputs: vec![],
        top_rank: 0,
        shape,
    }))
}


/// Creates a constant tensor.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let arr = ndarray::arr1(&[0., 0., 0.]);
/// let ref con = ag::constant(arr.clone(), &mut ctx);
/// assert_eq!(con.eval(&mut ctx), arr.into_dyn())
/// ```
#[inline]
pub fn constant<T>(arr: ndarray::Array<f32, T>, ctx: &mut Context) -> Tensor
where
    T: ndarray::Dimension,
{
    let arr = arr.into_dyn();
    let t = Tensor(Rc::new(RawTensor {
        op: Box::new(DummyOp { name: "Const".to_string() }),
        inputs: vec![],
        top_rank: 0,
        shape: Some(convert_to_tensor(::ndarray_ext::shape_of(&arr))),
    }));
    ctx.variables.insert(t.clone(), arr);
    t
}


/// Returns the (symbolic) shape of input tensor
///
/// ```
/// extern crate autograd as ag;
///
/// let ref x = ag::zeros(&[2, 3]);
/// let ref s = ag::shape(x);
///
/// let mut ctx = ag::Context::new();
/// assert_eq!(&[2., 3.], s.eval(&mut ctx).as_slice().unwrap());
/// ```
pub fn shape(x: &Tensor) -> Tensor
{
    if let Some(ref inner) = x.shape {
        inner.clone()
    } else {
        apply_op(array_ops::Shape, &[x], None)
    }
}


/// Returns the (symbolic) size of input tensor
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[4, 3]);
/// let ref b = ag::size(a);
///
/// let mut ctx = ag::Context::new();
/// assert_eq!(12., b.eval(&mut ctx)[ndarray::IxDyn(&[])]);
/// ```
pub fn size(x: &Tensor) -> Tensor
{
    apply_op(array_ops::Size, &[x], None)
}


/// Returns the (symbolic) rank of input tensor
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::zeros(&[2, 3, 4]);
/// let ref r = ag::rank(x);
///
/// let mut ctx = ag::Context::new();
/// assert_eq!(3., r.eval(&mut ctx)[ndarray::IxDyn(&[])]);
/// ```
pub fn rank(x: &Tensor) -> Tensor
{
    apply_op(array_ops::Rank, &[x], None)
}


/// Elementwise sine
pub fn sin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sin, &[x], Some(x.shape()))
}


/// Elementwise cosine
pub fn cos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cos, &[x], Some(x.shape()))
}


/// Elementwise tangent
pub fn tan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tan, &[x], Some(x.shape()))
}


/// Elementwise arcsin
pub fn asin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asin, &[x], Some(x.shape()))
}


/// Elementwise arccos
pub fn acos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acos, &[x], Some(x.shape()))
}


/// Elementwise arctan
pub fn atan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atan, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic sine
pub fn sinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sinh, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic cosine
pub fn cosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cosh, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic tangent
pub fn tanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tanh, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic arcsin
pub fn asinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asinh, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic arccos
pub fn acosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acosh, &[x], Some(x.shape()))
}


/// Elementwise hyperbolic arctan
pub fn atanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atanh, &[x], Some(x.shape()))
}


/// Identity function
pub fn identity(x: &Tensor) -> Tensor
{
    apply_op(activation_ops::Identity, &[x], Some(x.shape()))
}


/// Elementwise addition
///
/// `+` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::ones(&[3]);
/// let ref b = ag::ones(&[3]);
/// let ref z: ag::Tensor = a + b;
/// assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[2., 2., 2.]).into_dyn());
/// ```
pub fn add(a: &Tensor, b: &Tensor) -> Tensor
{
    let ref a_shape = a.shape();
    let ref b_shape = b.shape();
    apply_op(binary_ops::AddOp, &[a, b], Some(maximum(a_shape, b_shape)))
}


/// Elementwise subtraction
///
/// `-` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[3]);
/// let ref b = ag::ones(&[3]);
///
/// let mut ctx = ag::Context::new();
/// let ref z: ag::Tensor = a - b;
/// assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[0., 0., 0.]).into_dyn());
/// ```
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor
{
    let ref a_shape = a.shape();
    let ref b_shape = b.shape();
    apply_op(binary_ops::SubOp, &[a, b], Some(maximum(a_shape, b_shape)))
}


/// Elementwise multiplication
///
/// `*` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let ref a = ag::ones(&[3]);
/// let ref b = ag::ones(&[3]);
/// let ref z: ag::Tensor = a * b;
/// assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[1., 1., 1.]).into_dyn());
/// ```
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor
{
    let ref a_shape = a.shape();
    let ref b_shape = b.shape();
    apply_op(binary_ops::MulOp, &[a, b], Some(maximum(a_shape, b_shape)))
}


/// Elementwise division
///
/// `/` operator can be used instead.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::ones(&[3]);
/// let ref b = ag::ones(&[3]);
/// let mut ctx = ag::Context::new();
/// let ref z: ag::Tensor = a / b;
/// assert_eq!(z.eval(&mut ctx), ndarray::arr1(&[1., 1., 1.]).into_dyn());
/// ```
pub fn div(a: &Tensor, b: &Tensor) -> Tensor
{
    let ref a_shape = a.shape();
    let ref b_shape = b.shape();
    apply_op(binary_ops::DivOp, &[a, b], Some(maximum(a_shape, b_shape)))
}


/// Inplace addition
///
/// Returns `a` after performing `a += b`.
/// This function requires the move of `a`.
///
/// # Panics
///
/// When `a` is `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::add_inplace(a, b);
///
/// assert_eq!(c.eval(&mut ctx), ndarray::arr2(&[[2., 2.], [2., 2.]]).into_dyn());
/// ```
pub fn add_inplace(a: Tensor, b: &Tensor) -> Tensor
{
    assert_ne!(a.op.name(), "Const");
    let shape = a.shape();
    apply_op(binary_ops::InplaceAddOp, &[&a, b], Some(shape))
}


/// Inplace subtraction
///
/// Returns `a` after performing `a -= b`.
/// This function requires the move of `a`.
///
/// # Panics
///
/// When `a` is `constant`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::sub_inplace(a, b);
///
/// assert_eq!(c.eval(&mut ctx), ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn());
/// ```
pub fn sub_inplace(a: Tensor, b: &Tensor) -> Tensor
{
    assert_ne!(a.op.name(), "Const");
    let shape = a.shape();
    apply_op(binary_ops::InplaceSubOp, &[&a, b], Some(shape))
}


/// Elementwise sqrt
pub fn sqrt(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sqrt, &[x], Some(x.shape()))
}


/// Elementwise pow
pub fn pow(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Pow { a }, &[x], Some(x.shape()))
}


/// Elementwise log
pub fn log(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Log { a }, &[x], Some(x.shape()))
}


/// Elementwise exponential
pub fn exp(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Exp, &[x], Some(x.shape()))
}


/// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]), &mut ctx);
/// let ref c = ag::maximum(a, b);
/// assert_eq!(c.eval(&mut ctx), ndarray::arr1(&[3., 2., 3.]).into_dyn());
/// ```
pub fn maximum(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::Maximum, &[a, b], None)
}


/// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]), &mut ctx);
/// let ref c = ag::minimum(a, b);
/// assert_eq!(c.eval(&mut ctx), ndarray::arr1(&[1., 2., 1.]).into_dyn());
/// ```
pub fn minimum(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::Minimum, &[a, b], None)
}


/// Adds all input tensors.
///
/// All the input tensors must have same shapes.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::ones(&[2, 2]);
/// let ref d = ag::add_n(&[a, b, c]);
///
/// assert_eq!(d.eval(&mut ctx).shape(), &[2, 2]);
/// assert_eq!(d.eval(&mut ctx), ndarray::arr2(&[[3., 3.], [3., 3.]]).into_dyn());
/// ```
pub fn add_n(xs: &[&Tensor]) -> Tensor
{
    let len = xs.len();
    assert_ne!(len, 0);
    if len == 1 {
        xs[0].clone()
    } else {
        apply_op(array_ops::AddN, xs, Some(xs[0].shape()))
    }
}


/// Compares two tensors and returns a binary tensor.
///
/// if `a[i] == b[i]` then `return_value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]), &mut ctx);
/// let ref c = ag::equal(a, b);
///
/// assert_eq!(c.eval(&mut ctx), ndarray::arr1(&[0., 1., 0.]).into_dyn());
/// ```
pub fn equal(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::Equal, &[a, b], None)
}


/// Compares two tensors and returns a binary tensor.
///
/// if `a[i] != b[i]` then `return_value[i]` will be 1 else 0
///
/// # Panics
/// When broadcast is impossible
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[1., 2., 3.]), &mut ctx);
/// let ref b = ag::constant(ndarray::arr1(&[3., 2., 1.]), &mut ctx);
/// let ref c = ag::not_equal(a, b);
///
/// assert_eq!(c.eval(&mut ctx), ndarray::arr1(&[1., 0., 1.]).into_dyn());
/// ```
pub fn not_equal(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::NotEqual, &[a, b], None)
}


/// Takes argmax along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let input_arr = ndarray::arr2(&[[1., 2.], [3., 4.], [6., 5.]]);
/// let answer = ndarray::arr1(&[1., 1., 0.]).into_dyn();
/// let ref input = ag::constant(input_arr, &mut ctx);
/// let ref result = ag::argmax(&input, 1, false);
///
/// assert_eq!(result.eval(&mut ctx), answer);
/// ```
pub fn argmax(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ArgMax { axis, keep_dim };
    apply_op(op, &[x], None)
}


/// Expands specified dims.
///
/// Each axis can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[3]);
/// let ref b = ag::expand_dims(a, &[0, 2]);
///
/// assert_eq!(b.eval(&mut ctx).shape(), &[1, 3, 1]);
/// ```
pub fn expand_dims<T: ArrayLike>(x: &Tensor, axes: &T) -> Tensor
{
    apply_op(array_ops::ExpandDims, &[x, &axes.as_tensor()], None)
}


/// Squeezes specified dims.
///
/// Each axis can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[1, 3, 1]);
/// let ref b = ag::squeeze(a, &[0, 2]);
///
/// assert_eq!(b.eval(&mut ctx).shape(), &[3]);
/// ```
pub fn squeeze<T: ArrayLike>(x: &Tensor, axes: &T) -> Tensor
{
    apply_op(array_ops::Squeeze, &[x, &axes.as_tensor()], None)
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
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 2.], [3., 3.]]), &mut ctx);
/// let ref y = ag::tile(x, 0, 2);
///
/// assert_eq!(
///     y.eval(&mut ctx),
///     ndarray::arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn()
/// );
/// ```
pub fn tile(x: &Tensor, axis: isize, num: usize) -> Tensor
{
    let op = array_ops::Tile { axis, num };
    apply_op(op, &[x], None)
}


/// Limits all elements so as to be within `[min, max]`
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr1(&[2., 4., 6.]), &mut ctx);
/// let ref y = ag::clip(x, 3., 5.);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[3., 4., 5.]).into_dyn());
/// ```
pub fn clip(x: &Tensor, min: f32, max: f32) -> Tensor
{
    let op = array_ops::Clip { min, max };
    apply_op(op, &[x], Some(x.shape()))
}


/// Takes max along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]), &mut ctx);
/// let ref y = ag::reduce_max(&x, &[0], false);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[3., 4.]).into_dyn());
/// ```
pub fn reduce_max<T: ArrayLike>(x: &Tensor, axes: &T, keep_dims: bool) -> Tensor
{
    let op = reduction_ops::ReduceMax { keep_dims, sparse_axes: false };
    apply_op(op, &[x, &axes.as_tensor()], None)
}


/// Takes min along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]), &mut ctx);
/// let ref y = ag::reduce_min(&x, &[0], false);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[2., 1.]).into_dyn());
/// ```
pub fn reduce_min<T: ArrayLike>(x: &Tensor, axes: &T, keep_dims: bool) -> Tensor
{
    let op = reduction_ops::ReduceMin { keep_dims, sparse_axes: false };
    apply_op(op, &[x, &axes.as_tensor()], None)
}


/// Takes sum along specified axes.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]), &mut ctx);
/// let ref y = ag::reduce_sum(&x, &[1], false);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[6., 4.]).into_dyn());
/// ```
pub fn reduce_sum<T: ArrayLike>(x: &Tensor, axes: &T, keep_dims: bool) -> Tensor
{
    let ref axes = axes.as_tensor();
    let sum = reduction_ops::ReduceSum { keep_dims, sparse_axes: false };
    apply_op(sum, &[x, axes], None)
}


/// Takes mean along specified axis.
///
/// Elements of `axes` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]), &mut ctx);
/// let ref y = ag::reduce_mean(x, &[1], false);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[3., 2.]).into_dyn());
/// ```
pub fn reduce_mean<T: ArrayLike>(x: &Tensor, axes: &T, keep_dims: bool) -> Tensor
{
    let axes = rectify_negative_axes(&axes.as_tensor(), &x.rank());
    let op = reduction_ops::ReduceMean { keep_dims, sparse_axes: false };
    apply_op(op, &[x, &axes], None)
}


#[inline]
fn rectify_negative_axes(axes: &Tensor, x_rank: &Tensor) -> Tensor
{
    let ref zero = zeros(&axes.shape());
    let pos = greater_equal(axes, zero) * axes; // []
    let neg = lesser(axes, zero) * (axes + x_rank); // [1]
    pos + neg
}


/// Takes product along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::constant(ndarray::arr2(&[[2., 4.], [3., 1.]]), &mut ctx);
/// let ref y = ag::reduce_prod(&x, &[1], false);
///
/// assert_eq!(y.eval(&mut ctx), ndarray::arr1(&[8., 3.]).into_dyn());
/// ```
pub fn reduce_prod<T: ArrayLike>(x: &Tensor, axes: &T, keep_dims: bool) -> Tensor
{
    let op = reduction_ops::ReduceProd { keep_dims, sparse_axes: false };
    apply_op(op, &[x, &axes.as_tensor()], None)
}


/// Reshapes input tensor.
///
/// Only one dim in `shape` can be `-1`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::zeros(&[3, 2, 2]);
/// let ref y = ag::reshape(&x, &[3, -1]);
///
/// assert_eq!(y.eval(&mut ctx), ag::ndarray_ext::zeros(&[3, 4]));
/// ```
pub fn reshape<T: ArrayLike>(x: &Tensor, shape: &T) -> Tensor
{
    apply_op(array_ops::Reshape, &[x, &shape.as_tensor()], None)
}


/// Flattens input tensor into 1-ranked (vector)
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref x = ag::zeros(&[3, 2, 2]);
/// let ref z = ag::flatten(x);
/// assert_eq!(z.eval(&mut ctx).shape(), &[12]);
/// ```
pub fn flatten(x: &Tensor) -> Tensor
{
    apply_op(array_ops::Reshape, &[x, &scalar(-1.)], Some(x.size()))
}


/// Returns -1 if x < 0, 0 if x==0, 1 if x > 0, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[-5., 4.5, 0.]), &mut ctx);
/// let ref b = ag::sign(a);
/// assert_eq!(
///     b.eval(&mut ctx).as_slice().unwrap(),
///     &[-1., 1., 0.]
/// );
/// ```
pub fn sign(a: &Tensor) -> Tensor
{
    apply_op(math_ops::Sign, &[a], Some(a.shape()))
}


/// Returns the largest integer less than or equal to a number, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]), &mut ctx);
/// let ref b = ag::floor(a);
/// assert_eq!(
///     b.eval(&mut ctx).as_slice().unwrap(),
///     &[-2., -2., -1.,  0.,  1.,  1.,  2.]
/// );
/// ```
pub fn floor(a: &Tensor) -> Tensor
{
    apply_op(math_ops::Floor, &[a], Some(a.shape()))
}


/// Returns the 1/x, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[2., 3.]), &mut ctx);
/// let ref b = ag::square(a);
/// assert_eq!(
///     b.eval(&mut ctx).as_slice().unwrap(),
///     &[4., 9.]
/// );
/// ```
pub fn square(a: &Tensor) -> Tensor
{
    apply_op(math_ops::Square, &[a], Some(a.shape()))
}


/// Returns the 1/x, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[2.]), &mut ctx);
/// let ref b = ag::reciprocal(a);
/// assert_eq!(
///     b.eval(&mut ctx).as_slice().unwrap(),
///     &[0.5]
/// );
/// ```
pub fn reciprocal(a: &Tensor) -> Tensor
{
    apply_op(math_ops::Reciprocal, &[a], Some(a.shape()))
}


/// Returns the smallest integer greater than or equal to a number, element-wise.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]), &mut ctx);
/// let ref b = ag::ceil(a);
/// assert_eq!(
///     b.eval(&mut ctx).as_slice().unwrap(),
///     &[-1., -1., -0.,  1.,  2.,  2.,  2.]
/// );
/// ```
pub fn ceil(a: &Tensor) -> Tensor
{
    apply_op(math_ops::Ceil, &[a], Some(a.shape()))
}


/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::Greater, &[a, b], None)
}


/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn greater_equal(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::GreaterEqual, &[a, b], None)
}


/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::Lesser, &[a, b], None)
}


/// Returns a binary tensor.
///
/// # Panics
/// When broadcast is impossible
pub fn lesser_equal(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(math_ops::LesserEqual, &[a, b], None)
}


/// Elementwise logistic sigmoid function.
pub fn sigmoid(x: &Tensor) -> Tensor
{
    apply_op(activation_ops::Sigmoid, &[x], Some(x.shape()))
}


/// Elementwise exponential linear unit.
///
/// See https://arxiv.org/abs/1511.07289
pub fn elu(x: &Tensor, alpha: f32) -> Tensor
{
    apply_op(activation_ops::ELU { alpha }, &[x], Some(x.shape()))
}


/// Elementwise rectified linear unit.
pub fn relu(x: &Tensor) -> Tensor
{
    apply_op(activation_ops::ReLU, &[x], Some(x.shape()))
}


/// Computes `log(sum(exp(x)))` along specified axis.
pub fn logsumexp(x: &Tensor, axis: isize) -> Tensor
{
    let op = math_ops::LogSumExp { axis };
    apply_op(op, &[x], None)
}


/// Log softmax function.
///
/// Computes `softmax(x)` along specified axis and
/// takes logarithm of it.
/// `axis` can be negative.
pub fn log_softmax(x: &Tensor, axis: isize) -> Tensor
{
    // TODO: Composing from "node level" LogSumExp.
    let op = xent_ops::LogSoftmax { axis };
    apply_op(op, &[x], None)
}


/// Computes softmax along specified axis
///
/// `axis` can be negative.
pub fn softmax(x: &Tensor, axis: isize) -> Tensor
{
    let op = activation_ops::Softmax { axis };
    apply_op(op, &[x], None)
}


/// Computes `binary_cross_entropy(sigmoid(y), t)`.
///
/// This function is better than that combination in that it can prevent
/// underflow of `log(sigmoid)`.
///
/// # Arguments
/// * `y` - Tensor with arbitrary shape
/// * `t` - Tensor with arbitrary shape
///
/// # Panics
/// When y.shape != t.shape.
///
/// # Returns
/// Loss tensor with same shape as inputs's shapes
pub fn sigmoid_cross_entropy(y: &Tensor, t: &Tensor) -> Tensor
{
    apply_op(xent_ops::SigmoidCrossEntropy, &[y, t], Some(y.shape()))
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
pub fn softmax_cross_entropy(y: &Tensor, t: &Tensor) -> Tensor
{
    apply_op(
        xent_ops::SoftmaxCrossEntropyLatter,
        &[&log_softmax(y, 1), t],
        None,
    )
}


/// A variant of `softmax_cross_entropy`.
///
/// The behavior of this function is same as `softmax_cross_entropy`
/// except that `t` is **not** batch of one-hot distributions but batch of ground truth label ids.
///
/// # Arguments
/// * `y` - Tensor with shape (batch_size, num_classes)
/// * `t` - Tensor with shape (batch_size, 1)
///
/// # Returns
/// Loss tensor with shape (batch_size, 1)
pub fn sparse_softmax_cross_entropy(y: &Tensor, t: &Tensor) -> Tensor
{
    apply_op(
        xent_ops::SparseSoftmaxCrossEntropyLatter,
        &[&log_softmax(y, 1), t],
        None,
    )
}


/// Matrix multiplication.
///
/// Both `a` and `b` must be 2-ranked tensors.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
///
/// let ref a = ag::zeros(&[4, 2]);
/// let ref b = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul(a, b);
///
/// assert_eq!(c.eval(&mut ctx).shape(), &[4, 3]);
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = dot_ops::MatMul { transpose_a: false, transpose_b: false };
    apply_op(op, &[a, b], None)
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
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[2, 4]);
/// let ref b = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul_t(a, b, true, false);
///
/// assert_eq!(c.eval(&mut ctx).shape(), &[4, 3]);
/// ```
pub fn matmul_t(a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Tensor
{
    let op = dot_ops::MatMul { transpose_a, transpose_b };
    apply_op(op, &[a, b], None)
}


/// Computes tensor dot product (tensor contraction) along specified axes.
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
/// let mut ctx = ag::Context::new();
///
/// let ref a = ag::zeros(&[3, 4, 5]);
/// let ref b = ag::zeros(&[4, 3, 2]);
/// let ref c = ag::tensordot(a, b, &[1, 0], &[0, 1]);
/// assert_eq!(c.eval(&mut ctx).shape(), &[5, 2]);
/// ```
///
/// For detailed description,
/// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html.
pub fn tensordot<T: ArrayLike>(a: &Tensor, b: &Tensor, a_axes: &T, b_axes: &T) -> Tensor
{
    fn preprocess<T: ArrayLike>(x: &Tensor, axes: &T, flip: bool) -> (Tensor, Tensor)
    {
        let ref x_shape = x.shape();
        let ref x_rank = x.rank();
        let ref axes = rectify_negative_axes(&axes.as_tensor(), x_rank);
        let ref free = setdiff1d(&range(&scalar(0.), x_rank, &scalar(1.)), axes);

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
    let ((a_reshaped, a_free_dims), (b_reshaped, b_free_dims)) =
        (preprocess(a, a_axes, false), preprocess(b, b_axes, true));
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
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[2, 3, 4, 2]);
/// let ref b = ag::zeros(&[2, 3, 2, 3]);
/// let ref c = ag::batch_matmul(a, b);
///
/// assert_eq!(c.eval(&mut ag::Context::new()).shape(), &[2, 3, 4, 3]);
/// ```
///
/// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = dot_ops::BatchMatMul { transpose_a: false, transpose_b: false };
    apply_op(op, &[a, b], None)
}


/// Takes diff between two tensors
///
/// Returns the sorted, unique values in a that are not in b.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::constant(ndarray::arr1(&[4., 1., 5., 2., 3., 6.]), &mut ctx);
/// let ref b = ag::constant(ndarray::arr2(&[[2., 3.], [1., 4.]]), &mut ctx);
/// let ref c = ag::setdiff1d(a, b);
///
/// assert_eq!(c.eval(&mut ctx).as_slice().unwrap(), &[5., 6.])
/// ```
///
pub fn setdiff1d(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = array_ops::SetDiff1D;
    apply_op(op, &[a, b], None)
}


/// Permutes dimensions.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[1, 2, 3, 4, 5]);
/// let ref b = ag::transpose(a, &[4, 2, 3, 0, 1]);
///
/// assert_eq!(b.eval(&mut ctx).shape(), &[5, 3, 4, 1, 2]);
/// ```
pub fn transpose<T: ArrayLike>(x: &Tensor, perm: &T) -> Tensor
{
    let op = math_ops::Transpose { zip: true };
    apply_op(op, &[x, &perm.as_tensor()], None)
}


/// Splits input tensors into parts.
///
/// Splits `x` into `sizes.len()` parts along `axis`.
///
/// The size of dimension of each part is `sizes[i]` on `axis`, but
/// `x.shape[i]` on other axis.
///
/// ```
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[3, 7, 5]);
/// let ref b = ag::split(a, &[2, 3, 2], 1);
///
/// let evaluated = ag::eval(&[&b[0], &b[1], &b[2]], &mut ag::Context::new());
///
/// assert_eq!(evaluated[0].shape(), &[3, 2, 5]);
/// assert_eq!(evaluated[1].shape(), &[3, 3, 5]);
/// assert_eq!(evaluated[2].shape(), &[3, 2, 5]);
/// ```
pub fn split(x: &Tensor, sizes: &[usize], axis: isize) -> Vec<Tensor>
{
    (0..sizes.len())
        .map(|i| {
            let op = array_ops::Split { sizes: sizes.to_vec(), index: i, axis };
            apply_op(op, &[x], None)
        })
        .collect::<Vec<_>>()
}


/// Slices input tensor with indices.
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
/// let ref a = ag::zeros(&[4, 4]);
/// let ref b = ag::slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
///
/// assert_eq!(b.eval(&mut ag::Context::new()).shape(), &[4, 2]);
/// ```
pub fn slice(x: &Tensor, starts: &[isize], ends: &[isize]) -> Tensor
{
    // TODO: Make starts and ends ArrayLike
    assert_eq!(starts.len(), ends.len());
    let starts_ends = starts.iter().zip(ends.iter());

    let indices = starts_ends
        .map(|(s, e)| {
            ndarray::Si(*s, if *e == -1 { None } else { Some(*e) }, 1)
        })
        .collect::<Vec<ndarray::Si>>();

    let op = array_ops::Slice { indices: indices.into_boxed_slice() };

    apply_op(op, &[x], None)
}


/// Concatenates input tensors along specified axis.
///
/// `axis` can be negative.
///
/// ```
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let ref a = ag::zeros(&[3, 2]);
/// let ref b = ag::zeros(&[3, 2]);
/// let ref c = ag::zeros(&[3, 2]);
/// let ref d = ag::concat(&[a, b, c], 0);
///
/// assert_eq!(d.eval(&mut ctx).shape(), &[9, 2]);
/// ```
pub fn concat(tensors: &[&Tensor], axis: isize) -> Tensor
{
    apply_op(array_ops::Concat { axis }, tensors, None)
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
/// let mut ctx = ag::Context::new();
/// let ref param = ag::constant(ag::ndarray_ext::zeros(&[5, 4, 8, 2]), &mut ctx);
/// let ref indices = ag::constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]), &mut ctx);
/// let ref y = ag::gather(param, indices, 2);
///
/// assert_eq!(y.eval(&mut ctx).shape(), &[5, 4, 2, 3, 2])
/// ```
pub fn gather<T: ArrayLike>(param: &Tensor, indices: &T, axis: isize) -> Tensor
{
    let op = array_ops::Gather { axis };
    apply_op(op, &[&indices.as_tensor(), param], None)
}


/// Normalizes input tensor with its mean and variance along specified axis.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::standard_normal(&[3, 4]);
/// let ref y1 = ag::normalize(x, &[0]);
/// let ref y2 = ag::normalize(x, &[0]);
///
/// let evaluated = ag::eval(&[y1, y2], &mut ag::Context::new());
/// assert_eq!(&[3, 4], evaluated[0].shape());
/// assert_eq!(&[3, 4], evaluated[1].shape());
/// ```
pub fn normalize<T: ArrayLike>(x: &Tensor, axes: &T) -> Tensor
{
    let axes = axes.as_tensor();
    let ref mean = reduce_mean(x, &axes, true);
    let ref centered = x - mean;
    let ref variance = reduce_mean(&(centered * centered), &axes, true);
    (x - mean) / sqrt(&(variance + 1e-5))
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
/// let mut ctx = ag::Context::new();
/// let ref x = ag::standard_normal(&[3, 4]);
/// let ref scale = ag::variable(ag::ndarray_ext::ones(&[1, 4]), &mut ctx);
/// let ref shift = ag::variable(ag::ndarray_ext::zeros(&[1, 4]), &mut ctx);
/// let ref norm = ag::batch_norm(x, scale, shift);
///
/// assert_eq!(norm.eval(&mut ctx).shape(), &[3, 4]);
/// ```
pub fn batch_norm(x: &Tensor, scale: &Tensor, shift: &Tensor) -> Tensor
{
    normalize(x, &[0]) * scale + shift
}


/// Generates a zero-ranked tensor from a scalar value.
pub fn scalar(val: f32) -> Tensor
{
    apply_op(
        const_gen_ops::Scalar { val },
        &[],
        Some(convert_to_tensor(::ndarray_ext::scalar_shape())),
    )
}


/// Outputs values sampled from the normal distribution.
pub fn random_normal<T: ArrayLike>(shape: &T, mean: f64, stddev: f64) -> Tensor
{
    let op = random_ops::RandomNormal { mean, stddev };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the uniform distribution.
pub fn random_uniform<T: ArrayLike>(shape: &T, min: f64, max: f64) -> Tensor
{
    let op = random_ops::RandomUniform { min, max };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal<T: ArrayLike>(shape: &T) -> Tensor
{
    let op = random_ops::StandardNormal;
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform<T: ArrayLike>(shape: &T) -> Tensor
{
    let op = random_ops::StandardUniform;
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli<T: ArrayLike>(shape: &T, p: f64) -> Tensor
{
    let op = random_ops::Bernoulli { p };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the exponential distribution.
pub fn random_exp<T: ArrayLike>(shape: &T, lambda: f64) -> Tensor
{
    let op = random_ops::Exponential { lambda };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the gamma distribution.
pub fn random_gamma<T: ArrayLike>(shape: &T, shape_param: f64, scale: f64) -> Tensor
{
    let op = random_ops::Gamma { shape_param, scale };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Outputs values sampled from the log-normal distribution.
pub fn log_normal<T: ArrayLike>(shape: &T, mean: f64, stddev: f64) -> Tensor
{
    let op = random_ops::LogNormal { mean, stddev };
    let shape = shape.as_tensor();
    apply_op(op, &[&shape.clone()], Some(shape))
}


/// Converts `ndarray::Array` to `ag::Tensor`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let arr = ndarray::arr1(&[2., 3.]);
/// let tensor = ag::convert_to_tensor(arr.clone());
/// assert_eq!(tensor.eval(&mut ag::Context::new()), arr.into_dyn());
/// ```
pub fn convert_to_tensor<T>(arr: ndarray::Array<f32, T>) -> Tensor
where
    T: ndarray::Dimension,
{
    let arr = arr.into_dyn();
    let shape = {
        let op = const_gen_ops::ConvertToTensor { arr: ::ndarray_ext::shape_of(&arr) };
        apply_op(op, &[], None)
    };
    apply_op(const_gen_ops::ConvertToTensor { arr }, &[], Some(shape))
}


/// Returns zeros with given shape
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::zeros(&[4, 2]);
/// assert_eq!(a.eval(&mut ag::Context::new()), ndarray::Array2::<f32>::zeros((4, 2)).into_dyn());
/// ```
pub fn zeros<T: ArrayLike>(shape: &T) -> Tensor
{
    apply_op(const_gen_ops::Zeros, &[&shape.as_tensor()], None)
}


/// Returns ones with given shape
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::ones(&[4, 2]);
/// assert_eq!(a.eval(&mut ag::Context::new()),
///                   ndarray::Array2::<f32>::from_elem((4, 2), 1.).into_dyn());
/// ```
pub fn ones<T: ArrayLike>(shape: &T) -> Tensor
{
    apply_op(const_gen_ops::Ones, &[&shape.as_tensor()], None)
}


/// Returns a range
///
/// Unlike `range`, inputs are symbolic tensors.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref start = ag::scalar(0.);
/// let ref end = ag::scalar(5.);
/// let ref step = ag::scalar(1.);
/// let ref z = ag::range(start, end, step);
///
/// assert_eq!(z.eval(&mut ag::Context::new()), ndarray::Array1::range(0., 5., 1.).into_dyn());
/// ```
pub fn range<T: ArrayLike>(start: &T, end: &T, step: &T) -> Tensor
{
    apply_op(
        const_gen_ops::Range,
        &[&start.as_tensor(), &end.as_tensor(), &step.as_tensor()],
        None,
    )
}
