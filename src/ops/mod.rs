extern crate ndarray;

use ndarray_ext::NdArray;
use std::cell::RefCell;
use std::rc::Rc;
use tensor::{RawTensor, Tensor};

#[doc(hidden)]
pub mod dummy_op;
pub mod random_ops;
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
mod slice;
mod sigmoid_cross_entropy;
mod softmax_cross_entropy;
mod gather;
mod sparse_softmax_cross_entropy;
mod matmul;
mod swap_axes;
mod reshape;
mod reduction_ops;
mod squeeze;
mod expand_dims;


/// Represents a operation node in a computation graph.
/// `Tensor` wraps trait-object of this.
pub trait Op {
    /// Name of this op
    fn name(&self) -> &str;

    /// Returns gradient for each input node by use of output gradient etc.
    ///
    /// # Arguments
    /// * `gy` - Gradient of output of this op
    /// * `inputs` - `Tensor` level representation of `compute::xs`
    /// * `output` - `Tensor` level representation of `compute`'s return value
    fn grad(&self, gy: &Tensor, inputs: &[&Tensor], output: &Tensor) -> Vec<Option<Tensor>>;

    /// Actually runs this op.
    /// num of inputs : N,
    /// num of outputs: 1
    fn compute(&mut self, xs: &[&NdArray], train: bool) -> NdArray;
}


#[inline]
fn apply_op<T: Op + 'static>(op: T, inputs: &[&Tensor]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(op),
        inputs: inputs.iter().map(|a| (*a).clone()).collect::<Vec<Tensor>>(),
        param: None,
        rank: inputs
            .iter()
            .map(|a| a.borrow().rank)
            .max()
            .map(|a| a + 1)
            .unwrap_or(0),
    })))
}


// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

#[inline]
/// Hyperbolic arcsin function
pub fn asinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asinh, &[x])
}


#[inline]
/// Hyperbolic arccos function
pub fn acosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acosh, &[x])
}


#[inline]
/// Hyperbolic arctan function
pub fn atanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atanh, &[x])
}


#[inline]
/// Hyperbolic sine function
pub fn sinh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sinh, &[x])
}


#[inline]
/// Hyperbolic cosine function
pub fn cosh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cosh, &[x])
}


#[inline]
/// Hyperbolic tangent function
pub fn tanh(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tanh, &[x])
}


#[inline]
/// Arcsin function
pub fn asin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Asin, &[x])
}


#[inline]
/// Arccos function
pub fn acos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Acos, &[x])
}


#[inline]
/// Arctan function
pub fn atan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Atan, &[x])
}


#[inline]
/// Sine function
pub fn sin(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sin, &[x])
}


#[inline]
/// Cosine function
pub fn cos(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Cos, &[x])
}


#[inline]
/// Tangent function
pub fn tan(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Tan, &[x])
}


#[inline]
/// Adds all inputs
pub fn add_n(xs: &[&Tensor]) -> Tensor
{
    apply_op(add_n::AddN, xs)
}


#[inline]
/// Identity function
pub fn identity(a: &Tensor) -> Tensor
{
    apply_op(identity::Identity, &[a])
}


#[inline]
/// Adds two tensors
pub fn add(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::ElementwiseAdd, &[a, b])
}


#[inline]
/// Subtracts `a` from `b`
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::ElementwiseSub, &[a, b])
}


#[inline]
/// Multiplies two tensors
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::ElementwiseMul, &[a, b])
}


#[inline]
/// Divides `a` with `b`
pub fn div(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(binary_ops::ElementwiseDiv, &[a, b])
}


#[inline]
/// Sqrt
pub fn sqrt(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Sqrt, &[x])
}


#[inline]
/// Pow
pub fn pow(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Pow { a: a }, &[x])
}


/// Log
#[inline]
pub fn log(x: &Tensor, a: f32) -> Tensor
{
    apply_op(math_ops::Log { a: a }, &[x])
}


/// Exponential
#[inline]
pub fn exp(x: &Tensor) -> Tensor
{
    apply_op(math_ops::Exp, &[x])
}


#[inline]
/// Returns binary tensor.
///
/// # Panics
/// When a.shape != b.shape.
pub fn equals(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(cmp_ops::Equals, &[a, b])
}


#[inline]
/// Takes argmax along specified axis.
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
pub fn expand_dims(x: &Tensor, axes: &[isize]) -> Tensor
{
    let mut axes = axes.to_vec();
    axes.sort();
    apply_op(expand_dims::ExpandDims { axes: axes }, &[x])
}


#[inline]
/// Squeezes dims.
pub fn squeeze(x: &Tensor, axes: &[isize]) -> Tensor
{
    let mut axes = axes.to_vec();
    axes.sort();
    apply_op(squeeze::Squeeze { axes: axes }, &[x])
}


#[inline]
/// Tiles input tensor along specified axis.
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
pub fn clip(x: &Tensor, min: f32, max: f32) -> Tensor
{
    let op = clip::Clip { min: min, max: max };
    apply_op(op, &[x])
}


#[inline]
/// Take max along specified axis.
pub fn reduce_max(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMax {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Take min along specified axis.
pub fn reduce_min(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMin {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Take mean along specified axis.
pub fn reduce_mean(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceMean {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}


#[inline]
/// Take sum along specified axis.
pub fn reduce_sum(x: &Tensor, axis: isize, keep_dim: bool) -> Tensor
{
    let op = reduction_ops::ReduceSum {
        axis: axis,
        keep_dim: keep_dim,
    };
    apply_op(op, &[x])
}

#[inline]
/// Returns gradient tensors wrt variables.
///
/// # Arguments
/// * `objective` - Target of differentiation.
/// * `variables` - Variable tensors with which differentiate `objective`.
/// * `initial_grad` - This is required "if objective is not a scalar". In most cases,
/// this is initialized with 1s.
///
/// # Returns
/// Symbolic gradient tensors corresponding to `variables` in the same order as `variables`
pub fn gradients(
    objective: &Tensor,
    variables: &[&Tensor],
    initial_grad: Option<&Tensor>,
) -> Vec<Tensor>
{
    ::topology::symbolic_gradients(objective, variables, initial_grad)
}


#[inline]
/// Reshapes input tensor.
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
/// Returns 1-ranked tensor (vector)
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
/// Swaps two axes.
///
/// Swap axis `a` and axis `b` of `x`.
pub fn swap_axes(x: &Tensor, a: isize, b: isize) -> Tensor
{
    apply_op(swap_axes::SwapAxes { a: a, b: b }, &[x])
}


#[inline]
/// Elementwise logistic sigmoid function.
pub fn sigmoid(x: &Tensor) -> Tensor
{
    apply_op(sigmoid::Sigmoid, &[x])
}


#[inline]
/// Elementwise exponential linear unit function.
/// (https://arxiv.org/abs/1511.07289)
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
/// Computes log(softmax(x)) along specified axis.
pub fn log_softmax(x: &Tensor, axis: isize) -> Tensor
{
    // TODO: Composing from "node level" LogSumExp.
    let op = log_softmax::LogSoftmax { axis: axis };
    apply_op(op, &[x])
}


#[inline]
/// Softmax function.
///
/// Take softmax along `axis`.
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
/// `a` and `b` must be 2-ranked tensors.
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(matmul::MatMul, &[a, b])
}


#[inline]
/// Slice op.
///
/// # Arguments
/// * `x` - Tensor with arbitrary shape.
/// * `starts` - Start indices for each dimensions
/// * `ends` - End indices for each dimensions. `-1` representing the last index is acceptable.
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
/// Concat input tensors.
pub fn concat(tensors: &[&Tensor], axis: usize) -> Tensor
{
    apply_op(concat::Concat { axis: axis }, tensors)
}


#[inline]
/// Gather slices.
///
/// For example, this can be used for vector lookup.
/// See https://www.tensorflow.org/api_docs/python/tf/gather.
///
/// # Arguments
/// * `param` - Target of slicing.
/// * `indices` - Index tensor with arbitrary shape.
/// * `axis` - Slices sub tensors along this axis.
///
/// # Returns
/// Tensor with shape `(param.shape[..axis] + indices.shape + param.shape[axis+1..])`
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
pub fn rnn_step<T>(x: &Tensor, rnn: &mut T, with_new_state: bool) -> Tensor
where
    T: ::nn_impl::rnn::RNN,
{
    if with_new_state {
        rnn.reset_state();
    }
    rnn.step(x)
}
