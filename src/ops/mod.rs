extern crate ndarray;

use graph::Graph;
use ndarray_ext::NdArray;
use std::rc::Rc;
use tensor::{RawTensor, Tensor};

#[doc(hidden)]
pub mod dummy_op;
mod generator_ops;
mod scalar;
mod setdiff1d;
mod shape_ops;
mod stop_gradients;
mod index;
mod random_ops;
mod clip;
mod add_n;
mod logsumexp;
mod log_softmax;
mod identity;
mod cmp_ops;
mod math_ops;
mod concat;
mod tile;
mod binary_ops;
mod softmax;
mod sigmoid;
mod elu;
mod relu;
mod split;
mod slice;
mod sigmoid_cross_entropy;
mod softmax_cross_entropy;
mod sparse_softmax_cross_entropy;
mod gather;
mod matmul;
mod batch_matmul;
mod reverse_axes;
mod transpose;
mod reshape;
mod reduction_ops;
mod squeeze;
mod expand_dims;


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
    /// Num of inputs : N,
    /// Num of outputs: 1
    #[allow(unused_variables)]
    fn compute(&self, xs: &[&NdArray], train: bool) -> NdArray
    {
        unimplemented!()
    }

    /// Actually runs this op.
    ///
    /// Inplace operators such as AddAssign override this.
    #[allow(unused_variables)]
    fn compute_inplace(&self, xs: &mut [&mut NdArray], train: bool)
    {
        unimplemented!()
    }

    /// Returns symbolic gradient for each input node by use of output gradient etc.
    ///
    /// # Arguments
    /// * `gy` - Symbolic representation of the gradient of `compute`'s return value
    /// * `inputs` - Symbolic representation of `compute::xs`
    /// * `output` - Symbolic representation of `compute`'s return value
    ///
    /// NOTE:
    /// The number of return values must be same as `inputs.len()`.
    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>;
}


impl Tensor {
    /// Gets a symbolic float32 element from a tensor.
    ///
    /// `idx` can be negative.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut graph = ag::Graph::new();
    /// let ref a = graph.variable(ndarray::arr2(&[[2., 3.], [4., 5.]]));
    /// let ref b = a.get(2);
    /// assert_eq!(graph.eval(&[b])[0][0], 4.);
    /// ```
    pub fn get(&self, idx: isize) -> Tensor
    {
        apply_op(index::IndexOp { index: idx }, &[self])
    }
}


#[doc(hidden)]
#[inline]
/// Helper function to generate a symbolic tensor
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[4, 2]);
/// let ref v = ag::zeros(&[2, 3]);
/// let ref b = ag::zeros(&[4, 3]);
/// let ref z = ag::matmul(a, v) + b;
/// let mut vars = [a, v, b, z];
/// // `sort_by_key` don't reverse the order of `a` and `v`
/// vars.sort_by_key(|a| a.top_rank);
/// assert!(vars == [a, v, b, z])
/// ```
pub fn apply_op<T: Op + 'static>(op: T, inputs: &[&Tensor]) -> Tensor
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
    }))
}


// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------


#[inline]
/// Returns gradient tensors wrt variables.
///
/// # Arguments
/// * `objectives` - Targets of differentiation.
/// * `variables` - Variable tensors with which differentiate `objectives`.
/// * `output_grads` - Optionals. These are already known gradients of `objectives`.
/// So the length must be same as `objectives`'s.
/// If **each objective is not a scalar**, you must pass the "Some" value. In most cases,
/// it is initialized with 1s.
///
/// # Returns
/// Symbolic gradient tensors corresponding to `variables` in the same order as `variables`
///
/// # Example1
/// Partial derivatives of `z = 2x^2 + 3y + 1`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.placeholder();
/// let ref y = graph.variable(ndarray::arr1(&[0.]));
/// let ref z = 2*x*x + 3*y + 1;
///
/// // dz/dy
/// let ref g1 = ag::gradients(&[z], &[y], &[None])[0];
/// // dz/dx
/// let ref g2 = ag::gradients(&[z], &[x], &[None])[0];
///
/// // ddz/dx (differentiates `z` again)
/// let ref gg = ag::gradients(&[g2], &[x], &[None])[0];
///
/// // evaluation of symbolic gradients
/// assert_eq!(3., graph.eval(&[g1])[0][0]);
/// assert_eq!(4., graph.eval(&[gg])[0][0]);
///
/// // dz/dx requires to fill the placeholder `x`
/// graph.feed(x, ndarray::arr1(&[2.]));
/// assert_eq!(8., graph.eval(&[g2])[0][0]);
///
/// ```
///
/// # Example2
/// The case where objective is not a scalar
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.variable(ag::ndarray_ext::zeros(&[4, 2]));
/// let ref b = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul(a, b);
/// let ref g = ag::gradients(&[c], &[a], &[Some(&ag::ones(&[4, 2]))])[0];
/// ```
pub fn gradients(
    objectives: &[&Tensor],
    variables: &[&Tensor],
    output_grads: &[Option<&Tensor>],
) -> Vec<Tensor>
{
    ::topology::symbolic_gradients(objectives, variables, output_grads)
}


#[inline]
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
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.variable(ag::ndarray_ext::standard_normal(&[4, 2]));
/// let ref b = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
/// let ref c = ag::matmul(a, b);
/// let ref j = ag::jacobians(c, &[a, b], 4*3, );
///
/// assert_eq!(graph.eval(&[&j[0]])[0].shape(), &[4*3, 4*2]);
/// assert_eq!(graph.eval(&[&j[1]])[0].shape(), &[4*3, 2*3]);
/// ```
pub fn jacobians(objective: &Tensor, variables: &[&Tensor], objective_len: usize) -> Vec<Tensor>
{
    // TODO: remove map
    let vec_vec = (0..objective_len as isize)
        .map(|i| {
            // For each scalar objective, computes gradients for all variables
            ::topology::symbolic_gradients(&[&objective.get(i)], variables, &[None])
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


#[inline]
/// (Experimental) Computes hessian vector product
///
/// `objectives` must be scalars.
pub fn _hessian_vector_product(
    objectives: &[&Tensor],
    variables: &[&Tensor],
    vectors: &[&Tensor],
) -> Vec<Tensor>
{
    let grads = ::topology::symbolic_gradients(objectives, variables, &[None]);

    let products = grads
        .iter()
        .zip(vectors)
        .map(|(g, &v)| g * v)
        .collect::<Vec<_>>();

    let products = products.iter().map(|a| a).collect::<Vec<_>>();

    ::topology::symbolic_gradients(products.as_slice(), variables, &[None])
}


#[inline]
/// Stops gradients
pub fn stop_gradients(x: &Tensor) -> Tensor
{
    apply_op(stop_gradients::StopGradients, &[x])
}


/// Returns symbolic shape of input tensor
///
/// This is useful when the shape of `x` is dynamic.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.placeholder();
/// let ref s = ag::shape(x);
///
/// graph.feed(x, ag::ndarray_ext::zeros(&[2, 3]));
/// assert_eq!(&[2., 3.], graph.eval(&[s])[0].as_slice().unwrap());
/// ```
pub fn shape(x: &Tensor) -> Tensor
{
    apply_op(shape_ops::Shape, &[x])
}


/// Returns the (symbolic) length of input tensor
///
/// This is useful when the size of `x` is dynamic.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.placeholder();
/// let ref b = ag::zeros(&[4, 3]);
/// let ref c = ag::size(a);
/// let ref d = ag::size(b);
///
/// graph.feed(a, ag::ndarray_ext::zeros(&[2, 3]));
/// assert_eq!(6., graph.eval(&[c])[0][0]);
/// assert_eq!(12., graph.eval(&[d])[0][0]);
/// ```
pub fn size(x: &Tensor) -> Tensor
{
    apply_op(shape_ops::Size, &[x])
}


/// Returns the (symbolic) rank of input tensor
///
/// This is useful when the rank of `x` is dynamic.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.placeholder();
/// let ref r = ag::rank(x);
///
/// graph.feed(x, ag::ndarray_ext::zeros(&[2, 3]));
/// assert_eq!(2., graph.eval(&[r])[0][0]);
/// ```
pub fn rank(x: &Tensor) -> Tensor
{
    apply_op(shape_ops::Rank, &[x])
}


#[inline]
/// Elementwise sine
pub fn sin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sin, &[x])
}


#[inline]
/// Elementwise cosine
pub fn cos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cos, &[x])
}


#[inline]
/// Elementwise tangent
pub fn tan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tan, &[x])
}


#[inline]
/// Elementwise arcsin
pub fn asin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asin, &[x])
}


#[inline]
/// Elementwise arccos
pub fn acos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acos, &[x])
}


#[inline]
/// Elementwise arctan
pub fn atan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atan, &[x])
}


#[inline]
/// Elementwise hyperbolic sine
pub fn sinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sinh, &[x])
}


#[inline]
/// Elementwise hyperbolic cosine
pub fn cosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cosh, &[x])
}


#[inline]
/// Elementwise hyperbolic tangent
pub fn tanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tanh, &[x])
}


#[inline]
/// Elementwise hyperbolic arcsin
pub fn asinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asinh, &[x])
}


#[inline]
/// Elementwise hyperbolic arccos
pub fn acosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acosh, &[x])
}


#[inline]
/// Elementwise hyperbolic arctan
pub fn atanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atanh, &[x])
}


#[inline]
/// Identity function
pub fn identity(a: &Tensor) -> Tensor
{
    apply_op(identity::Identity, &[a])
}


#[inline]
/// Element-wise addition
///
/// You can use `+` operator instead.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::AddOp, &[a, b])
}


#[inline]
/// Element-wise subtraction
///
/// You can use `-` operator instead.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::SubOp, &[a, b])
}


#[inline]
/// Element-wise multiplication
///
/// You can use `*` operator instead.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::MulOp, &[a, b])
}


#[inline]
/// Element-wise division
///
/// You can use `/` operator instead.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::DivOp, &[a, b])
}


#[inline]
/// Inplace addition
///
/// Returns a symbolic tensor for `a` after performing `a += b`
/// You can not use `a` after calling this function.
///
/// # Panics
///
/// When `a` is from `graph#constant` or `graph#variable`.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
///
/// let a = ag::zeros(&[2, 2]) + ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::add_inplace(a, b);
/// assert_eq!(graph.eval(&[c])[0], ndarray::arr2(&[[2., 2.], [2., 2.]]).into_dyn());
/// ```
pub fn add_inplace(a: Tensor, b: &Tensor) -> Tensor
{
    let a_name = a.op.name();
    assert!(a_name != "Constant" && a_name != "Variable");
    apply_op(binary_ops::InplaceAddOp, &[&a, b])
}


#[inline]
/// Inplace subtraction
///
/// Returns a symbolic tensor for `a` after performing `a -= b`
/// You can not use `a` after calling this function.
///
/// # Panics
///
/// When `a` is from `graph#constant` or `graph#variable`.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
///
/// let a = ag::zeros(&[2, 2]) + ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::sub_inplace(a, b);
/// assert_eq!(graph.eval(&[c])[0], ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn());
/// ```
pub fn sub_inplace(a: Tensor, b: &Tensor) -> Tensor
{
    let a_name = a.op.name();
    assert!(a_name != "Constant" && a_name != "Variable");
    apply_op(binary_ops::InplaceSubOp, &[&a, b])
}


#[inline]
/// Elementwise sqrt
pub fn sqrt(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sqrt, &[x])
}


#[inline]
/// Elementwise pow
pub fn pow(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Pow { a: a }, &[x])
}


/// Elementwise log
#[inline]
pub fn log(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Log { a: a }, &[x])
}


/// Elementwise exponential
#[inline]
pub fn exp(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Exp, &[x])
}


#[inline]
/// Adds all input tensors inplace.
///
/// All the input tensors must have same shapes.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::ones(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
/// let ref c = ag::ones(&[2, 2]);
/// let ref d = ag::add_n(&[a, b, c]);
/// assert_eq!(graph.eval(&[d])[0].shape(), &[2, 2]);
/// assert_eq!(graph.eval(&[d])[0], ndarray::arr2(&[[3., 3.], [3., 3.]]).into_dyn());
/// ```
pub fn add_n(xs: &[&Tensor]) -> Tensor
{
    apply_op(add_n::AddN, xs)
}


#[inline]
/// Compares two tensors and returns a binary tensor.
///
/// if `a[i] == b[i]` then `return_value[i]` will be 1 else 0
///
/// # Panics
/// When `a's shape` != `b's shape`.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.constant(ndarray::arr1(&[1., 2., 3.]));
/// let ref b = graph.constant(ndarray::arr1(&[3., 2., 1.]));
/// let ref c = ag::equals(a, b);
/// assert_eq!(graph.eval(&[c])[0], ndarray::arr1(&[0., 1., 0.]).into_dyn());
/// ```
pub fn equals(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(cmp_ops::Equals, &[a, b])
}


#[inline]
/// Takes argmax along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let input_arr = ndarray::arr2(&[[1., 2.], [3., 4.], [6., 5.]]);
/// let answer = ndarray::arr1(&[1., 1., 0.]).into_dyn();
/// let ref input = graph.constant(input_arr);
/// let ref result = ag::argmax(&input, 1, false);
/// assert_eq!(graph.eval(&[result])[0], answer);
/// ```
pub fn argmax(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ArgMax {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Expands dims.
///
/// Each axis can be negative.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.constant(ag::ndarray_ext::standard_normal(&[3]));
/// let ref b = ag::expand_dims(a, &[0, 2]);
/// assert_eq!(graph.eval(&[b])[0].shape(), &[1, 3, 1]);
/// ```
pub fn expand_dims(x: &Tensor, axes: &[isize]) -> Tensor
{
    let mut axes = axes.to_vec();
    axes.sort();
    apply_op(expand_dims::ExpandDims { axes: axes }, &[x])
}


#[inline]
/// Squeezes dims.
///
/// Each axis can be negative.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.constant(ag::ndarray_ext::standard_normal(&[1, 3, 1]));
/// let ref b = ag::squeeze(a, &[0, 2]);
/// assert_eq!(graph.eval(&[b])[0].shape(), &[3]);
/// ```
pub fn squeeze(x: &Tensor, axes: &[isize]) -> Tensor
{
    let mut axes = axes.to_vec();
    axes.sort();
    apply_op(squeeze::Squeeze { axes: axes }, &[x])
}


#[inline]
/// Tiles input tensor along specified axis.
///
/// Tiles input tensor `num` times along `axis`.
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 2.], [3., 3.]]));
/// let ref y = ag::tile(x, 0, 2);
/// assert_eq!(
///     graph.eval(&[y])[0],
///     ndarray::arr2(&[[2., 2.], [3., 3.], [2., 2.], [3., 3.]]).into_dyn()
/// );
/// ```
pub fn tile(x: &Tensor, axis: isize, num: usize) -> Tensor
{
    let op = tile::Tile {
        axis: axis,
        num: num,
    };
    apply_op(op, &[x])
}


#[inline]
/// Limits all elements so as to be within `[min, max]`
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr1(&[2., 4., 6.]));
/// let ref y = ag::clip(x, 3., 5.);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[3., 4., 5.]).into_dyn());
/// ```
pub fn clip(x: &Tensor, min: f32, max: f32) -> Tensor
{
    let op = clip::Clip { min: min, max: max };
    apply_op(op, &[x])
}


#[inline]
/// Takes max along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_max(&x, 0, false);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[3., 4.]).into_dyn());
/// ```
pub fn reduce_max(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMax {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Takes min along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_min(&x, 0, false);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[2., 1.]).into_dyn());
/// ```
pub fn reduce_min(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMin {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Takes mean along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_mean(x, 1, false);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[3., 2.]).into_dyn());
/// ```
pub fn reduce_mean(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMean {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Takes sum along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_sum(&x, 1, false);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[6., 4.]).into_dyn());
/// ```
pub fn reduce_sum(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceSum {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Takes product along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = graph.constant(ndarray::arr2(&[[2., 4.], [3., 1.]]));
/// let ref y = ag::reduce_prod(&x, 1, false);
/// assert_eq!(graph.eval(&[y])[0], ndarray::arr1(&[8., 3.]).into_dyn());
/// ```
pub fn reduce_prod(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceProd {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Reshapes input tensor.
///
/// Only one dim in `shape` can be `-1`.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = ag::zeros(&[3, 2, 2]);
/// let ref y = ag::reshape(&x, &[3, 4]);
/// assert_eq!(graph.eval(&[y])[0], ag::ndarray_ext::zeros(&[3, 4]));
/// ```
pub fn reshape(x: &Tensor, shape: &[isize]) -> Tensor
{
    let mut minus_one_found = false;
    let shape = shape
        .iter()
        .map(|&len| if len == -1 {
            if minus_one_found {
                panic!("`shape` has two or more `-1` dim.");
            }
            minus_one_found = true;
            None
        } else if len < -1 {
            panic!("`shape` contains invalid dim size: {}", len);
        } else {
            Some(len as usize)
        })
        .collect::<Vec<_>>();
    let op = reshape::Reshape { target_shape: shape };
    apply_op(op, &[x])
}


#[inline]
/// Flattens input tensor into 1-ranked (vector)
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x = ag::zeros(&[3, 2, 2]);
/// let ref z = ag::flatten(x);
/// assert_eq!(graph.eval(&[z])[0].shape(), &[12]);
/// ```
pub fn flatten(x: &Tensor) -> Tensor
{
    let op = reshape::Reshape { target_shape: vec![None] };
    apply_op(op, &[x])
}


#[inline]
/// Returns binary tensor.
pub fn greater(x: &Tensor, a: f32) -> Tensor
{
    apply_op(cmp_ops::Greater { a: a }, &[x])
}


#[inline]
/// Returns binary tensor.
pub fn greater_equal(x: &Tensor, a: f32) -> Tensor
{
    apply_op(cmp_ops::GreaterEqual { a: a }, &[x])
}


#[inline]
/// Returns binary tensor.
pub fn lesser(x: &Tensor, a: f32) -> Tensor
{
    apply_op(cmp_ops::Lesser { a: a }, &[x])
}


#[inline]
/// Returns binary tensor.
pub fn lesser_equal(x: &Tensor, a: f32) -> Tensor
{
    apply_op(cmp_ops::LesserEqual { a: a }, &[x])
}


#[inline]
/// Reverses axes of the input tensor.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.constant(ag::ndarray_ext::zeros(&[2, 3, 4, 5]));
/// let ref b = ag::reverse_axes(a);
/// assert_eq!(graph.eval(&[b])[0].shape(), &[5, 4, 3, 2]);
/// ```
pub fn reverse_axes(x: &Tensor) -> Tensor
{
    apply_op(reverse_axes::ReverseAxes, &[x])
}


#[inline]
/// Elementwise logistic sigmoid function.
pub fn sigmoid(x: &Tensor) -> Tensor
{
    apply_op(sigmoid::Sigmoid, &[x])
}


#[inline]
/// Elementwise exponential linear unit function.
///
/// See https://arxiv.org/abs/1511.07289
pub fn elu(x: &Tensor, alpha: f32) -> Tensor
{
    apply_op(elu::ELU { alpha: alpha }, &[x])
}


#[inline]
/// Elementwise rectified linear unit function.
pub fn relu(x: &Tensor) -> Tensor
{
    apply_op(relu::ReLU, &[x])
}


#[inline]
/// Computes `log(sum(exp(x)))` along specified axis.
pub fn logsumexp(x: &Tensor, axis: isize) -> Tensor
{
    let op = logsumexp::LogSumExp { axis: axis };
    apply_op(op, &[x])
}


#[inline]
/// Log softmax function.
///
/// Computes `softmax(x)` along specified axis and
/// takes logarithm of it.
/// `axis` can be negative.
pub fn log_softmax(x: &Tensor, axis: isize) -> Tensor
{
    // TODO: Composing from "node level" LogSumExp.
    let op = log_softmax::LogSoftmax { axis: axis };
    apply_op(op, &[x])
}


#[inline]
/// Takes softmax along specified axis
///
/// `axis` can be negative.
pub fn softmax(x: &Tensor, axis: isize) -> Tensor
{
    let op = softmax::Softmax { axis: axis };
    apply_op(op, &[x])
}


#[inline]
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
    let op = sigmoid_cross_entropy::SigmoidCrossEntropy;
    apply_op(op, &[y, t])
}


#[inline]
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
    let op = softmax_cross_entropy::SoftmaxCrossEntropy;
    apply_op(op, &[y, t])
}


#[inline]
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
    let op = sparse_softmax_cross_entropy::SparseSoftmaxCrossEntropy;
    apply_op(op, &[y, t])
}


#[inline]
/// Matrix multiplication.
///
/// Both `a` and `b` must be 2-ranked tensors.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
///
/// let ref a = ag::zeros(&[4, 2]);
/// let ref b = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul(a, b);
/// assert_eq!(graph.eval(&[c])[0].shape(), &[4, 3]);
/// ```
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = matmul::MatMul {
        transpose_a: false,
        transpose_b: false,
    };
    apply_op(op, &[a, b])
}


#[inline]
/// Matrix multiplication.
///
/// Similar specification as `matmul` but, if `transpose_a` is true, `a` is transposed
/// before actual matrix multiplication. It is the same for `transpose_b`.
///
/// The performance is better than explicitly computing like `ag::matmul(ag::transpose)`.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[2, 4]);
/// let ref b = ag::zeros(&[2, 3]);
/// let ref c = ag::matmul_t(a, b, true, false);
/// assert_eq!(graph.eval(&[c])[0].shape(), &[4, 3]);
/// ```
pub fn matmul_t(a: &Tensor, b: &Tensor, transpose_a: bool, transpose_b: bool) -> Tensor
{
    let op = matmul::MatMul {
        transpose_a: transpose_a,
        transpose_b: transpose_b,
    };
    apply_op(op, &[a, b])
}


#[inline]
/// Computes tensor dot product (tensor contraction) along specified axes.
///
/// # Arguments
/// * `a` - Input tensor
/// * `b` - Input tensor
/// * `a_shape` - Shape of a
/// * `b_shape` - Shape of b
/// * `axes` - `axes[0]` and `axes[1]` are `a`'s and `b`'s axes respectively.
/// Contraction is computed along corresponding `a`'s and `b`'s axes.
/// So the number of the axes must be equals.
///
/// Note: each axis can be negative number.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
///
/// let ref a = ag::zeros(&[3, 4, 5]);
/// let ref b = ag::zeros(&[4, 3, 2]);
/// let ref c = ag::tensordot(a, b, &[3, 4, 5], &[4, 3, 2], [&[1, 0], &[0, 1]]);
/// assert_eq!(graph.eval(&[c])[0].shape(), &[5, 2]);
///
/// // Another example (simple matmul broadcast)
/// let ref a = ag::zeros(&[2, 3, 4]);
/// let ref b = ag::zeros(&[4, 2]);
/// let ref c = ag::tensordot(a, b, &[2, 3, 4], &[4, 2], [&[2], &[0]]);
/// assert_eq!(graph.eval(&[c])[0].shape(), &[2, 3, 2]);
/// ```
///
/// For detailed description,
/// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html.
pub fn tensordot(
    a: &Tensor,
    b: &Tensor,
    a_shape: &[usize],
    b_shape: &[usize],
    axes: [&[isize]; 2],
) -> Tensor
{
    assert_eq!(axes[0].len(), axes[1].len());

    fn preprocess(x: &Tensor, x_shape: &[usize], axes: &[isize], flip: bool)
        -> (Tensor, Vec<isize>)
    {
        let axes = axes.iter()
            .map(|&i| if i >= 0 {
                i as usize
            } else {
                (i + x_shape.len() as isize) as usize
            })
            .collect::<Vec<_>>();

        let mut free: Vec<usize> = vec![];
        for i in 0..x_shape.len() {
            if !axes.contains(&i) {
                free.push(i);
            }
        }
        let free_dims = free.clone()
            .into_iter()
            .map(|i| x_shape[i] as isize)
            .collect::<Vec<_>>();
        let prod_free: isize = free_dims.clone().into_iter().product();
        let prod_axes: usize = axes.iter().cloned().map(|a| x_shape[a]).product();
        let perm = if flip {
            axes.into_iter().chain(free).collect::<Vec<_>>()
        } else {
            free.into_iter().chain(axes).collect::<Vec<_>>()
        };
        let new_shape = if flip {
            [prod_axes as isize, prod_free]
        } else {
            [prod_free, prod_axes as isize]
        };
        let reshaped = reshape(&transpose(x, perm.as_slice()), &new_shape);
        (reshaped, free_dims)
    }

    let (a_reshaped, a_free_dims) = preprocess(a, a_shape, axes[0], false);
    let (b_reshaped, b_free_dims) = preprocess(b, b_shape, axes[1], true);

    let ref dot = matmul(&a_reshaped, &b_reshaped);
    let final_shape = a_free_dims
        .into_iter()
        .chain(b_free_dims.into_iter())
        .collect::<Vec<isize>>();

    reshape(dot, final_shape.as_slice())
}


#[inline]
/// Batched matrix multiplication.
///
/// Performs matrix multiplication between corresponding dimensions of `a` and `b`.
/// So the rank of `a` and `b` must be equals.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[2, 3, 4, 2]);
/// let ref b = ag::zeros(&[2, 3, 2, 3]);
/// let ref c = ag::batch_matmul(a, b);
/// assert_eq!(graph.eval(&[c])[0].shape(), &[2, 3, 4, 3]);
/// ```
///
/// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = batch_matmul::BatchMatMul {
        transpose_a: false,
        transpose_b: false,
    };
    apply_op(op, &[a, b])
}


#[inline]
/// Takes diff between two tensor
///
/// Returns the sorted, unique values in a that are not in b.
///
/// # Examples
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = graph.constant(ndarray::arr1(&[4., 1., 5., 2., 3., 6.]));
/// let ref b = graph.constant(ndarray::arr2(&[[2., 3.], [1., 4.]]));
/// let ref c = ag::setdiff1d(a, b);
/// assert_eq!(graph.eval(&[c])[0].as_slice().unwrap(), &[5., 6.])
/// ```
///
pub fn setdiff1d(a: &Tensor, b: &Tensor) -> Tensor
{
    let op = setdiff1d::SetDiff1D;
    apply_op(op, &[a, b])
}


#[inline]
/// Permutes dimensions.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[1, 2, 3, 4, 5]);
/// let ref b = ag::transpose(a, &[4, 2, 3, 0, 1]);
/// assert_eq!(graph.eval(&[b])[0].shape(), &[5, 3, 4, 1, 2]);
/// ```
pub fn transpose(x: &Tensor, perm: &[usize]) -> Tensor
{
    let src_dst = perm.iter().cloned().zip(0..perm.len()).collect::<Vec<_>>();
    let op = transpose::Transpose { src_dst_sorted: src_dst };
    apply_op(op, &[x])
}


#[inline]
/// Splits input tensors into parts.
///
/// Splits `x` into `sizes.len()` parts along `axis`.
/// The size of dimension of each part is `sizes[i]` on `axis`, but
/// `x.shape[i]` on other axis.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[3, 7, 5]);
/// let ref b = ag::split(a, &[2, 3, 2], 1);
/// assert_eq!(graph.eval(&[&b[0]])[0].shape(), &[3, 2, 5]);
/// assert_eq!(graph.eval(&[&b[1]])[0].shape(), &[3, 3, 5]);
/// assert_eq!(graph.eval(&[&b[2]])[0].shape(), &[3, 2, 5]);
/// ```
pub fn split(x: &Tensor, sizes: &[usize], axis: isize) -> Vec<Tensor>
{
    (0..sizes.len())
        .map(|i| {
            let op = split::Split {
                sizes: sizes.to_vec(),
                index: i,
                axis: axis,
            };
            apply_op(op, &[x])
        })
        .collect::<Vec<_>>()
}


#[inline]
/// Slices input tensor with indices.
///
/// # Arguments
/// * `x` - Tensor with arbitrary shape.
/// * `starts` - Start indices for each dimensions
/// * `ends` - End indices for each dimensions. `-1` representing the last index is acceptable.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[4, 4]);
/// let ref b = ag::slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
/// assert_eq!(graph.eval(&[b])[0].shape(), &[4, 2]);
/// ```
pub fn slice(x: &Tensor, starts: &[isize], ends: &[isize]) -> Tensor
{
    assert_eq!(starts.len(), ends.len());
    let starts_ends = starts.iter().zip(ends.iter());

    let indices = starts_ends
        .map(|(s, e)| {
            ndarray::Si(*s, if *e == -1 { None } else { Some(*e) }, 1)
        })
        .collect::<Vec<ndarray::Si>>();

    let op = slice::Slice { indices: indices.into_boxed_slice() };

    apply_op(op, &[x])
}


#[inline]
/// Concatenates (stacks) input tensors along specified axis.
///
/// `axis` can be negative.
///
/// # Examples
///
/// ```
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref a = ag::zeros(&[3, 2]);
/// let ref b = ag::zeros(&[3, 2]);
/// let ref c = ag::zeros(&[3, 2]);
/// let ref d = ag::concat(&[a, b, c], 0);
/// assert_eq!(graph.eval(&[d])[0].shape(), &[9, 2]);
/// ```
pub fn concat(tensors: &[&Tensor], axis: isize) -> Tensor
{
    apply_op(concat::Concat { axis: axis }, tensors)
}


#[inline]
/// Gathers slices.
///
/// Along `axis`, slices subviews from `param` with `indices`, and then gathers those.
/// For example, this can be used for embedding vector lookup.
/// `axis` can be negative.
///
/// See also https://www.tensorflow.org/api_docs/python/tf/gather.
///
/// # Arguments
/// * `param` - Target of slicing.
/// * `indices` - Index tensor with which slices `param`. This can be arbitrary shape.
/// * `axis` - Slices sub tensors along this axis.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref param = graph.constant(ag::ndarray_ext::zeros(&[5, 4, 8, 2]));
/// let ref indices = graph.constant(ndarray::arr2(&[[5., 4., 3.], [2., 1., 0.]]));
/// let ref y = ag::gather(param, indices, 2);
/// assert_eq!(graph.eval(&[y])[0].shape(), &[5, 4, 2, 3, 2])
/// ```
pub fn gather(param: &Tensor, indices: &Tensor, axis: isize) -> Tensor
{
    let op = gather::Gather { axis: axis };
    apply_op(op, &[indices, param])
}


#[inline]
/// Applies recurrent net unit to the input.
///
/// This func processes a time step in the batch of sequences in parallel.
///
/// # Arguments
/// * `x` - Input tensor for this step
/// * `rnn` - RNN struct
/// * `with_new_state` - If true, calls `rnn.reset_state()` before running a step
///
/// # Returns
/// Output of `rnn.step()`
///
/// For the usage, see `lstm_lm()` in `tests/test_tensor_ops_grad.rs` and `nn_impl::rnn`
pub fn rnn_step<T>(x: &Tensor, rnn: &mut T, with_new_state: bool, g: &mut Graph) -> Tensor
where
    T: ::nn_impl::rnn::RNN,
{
    if with_new_state {
        rnn.reset_state(g);
    }
    rnn.step(x)
}



/// Creates a constant tensor.
#[inline]
pub fn scalar(val: f32) -> Tensor
{
    apply_op(scalar::Scalar { val: val }, &[])
}


/// Outputs values sampled from the normal distribution.
pub fn random_normal(shape: &[usize], mean: f64, stddev: f64) -> Tensor
{
    let op = random_ops::RandomNormal {
        shape: shape.to_vec(),
        mean: mean,
        stddev: stddev,
    };
    apply_op(op, &[])
}


/// Outputs values sampled from the uniform distribution.
pub fn random_uniform(shape: &[usize], min: f64, max: f64) -> Tensor
{
    let op = random_ops::RandomUniform {
        shape: shape.to_vec(),
        min: min,
        max: max,
    };
    apply_op(op, &[])
}


/// Outputs values sampled from the standard normal distribution.
pub fn standard_normal(shape: &[usize]) -> Tensor
{
    let op = random_ops::StandardNormal { shape: shape.to_vec() };
    apply_op(op, &[])
}


/// Outputs values sampled from the standard uniform distribution.
pub fn standard_uniform(shape: &[usize]) -> Tensor
{
    let op = random_ops::StandardUniform { shape: shape.to_vec() };
    apply_op(op, &[])
}


/// Outputs values sampled from the bernoulli distribution.
pub fn bernoulli(shape: &[usize], p: f64) -> Tensor
{
    let op = random_ops::Bernoulli {
        shape: shape.to_vec(),
        p: p,
    };
    apply_op(op, &[])
}


/// Outputs values sampled from the exponential distribution.
pub fn random_exp(shape: &[usize], lambda: f64) -> Tensor
{
    let op = random_ops::Exponential {
        shape: shape.to_vec(),
        lambda: lambda,
    };
    apply_op(op, &[])
}


/// Outputs values sampled from the gamma distribution.
pub fn gamma(shape: &[usize], shape_param: f64, scale: f64) -> Tensor
{
    let op = random_ops::Gamma {
        shape: shape.to_vec(),
        shape_param: shape_param,
        scale: scale,
    };
    apply_op(op, &[])
}


/// Outputs values sampled from the log-normal distribution.
pub fn log_normal(shape: &[usize], mean: f64, stddev: f64) -> Tensor
{
    let op = random_ops::LogNormal {
        shape: shape.to_vec(),
        mean: mean,
        stddev: stddev,
    };
    apply_op(op, &[])
}


#[inline]
/// Returns zeros with given shape
pub fn zeros(shape: &[usize]) -> Tensor
{
    let op = generator_ops::Zeros { shape: shape.to_vec() };
    apply_op(op, &[])
}


#[inline]
/// Returns ones with given shape
pub fn ones(shape: &[usize]) -> Tensor
{
    let op = generator_ops::Ones { shape: shape.to_vec() };
    apply_op(op, &[])
}


/// Returns range
#[inline]
pub fn range(start: usize, end: usize, step: usize) -> Tensor
{
    let op = generator_ops::Range {
        start: start as f32,
        end: end as f32,
        step: step as f32,
    };
    apply_op(op, &[])
}
