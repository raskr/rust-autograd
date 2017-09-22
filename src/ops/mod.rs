extern crate ndarray;

use ndarray_ext::NdArray;
use std::cell::RefCell;
use std::rc::Rc;
use tensor::{RawTensor, Tensor};

mod dummy_op;
pub mod random_ops;
pub mod clip;
pub mod add_n;
pub mod logsumexp;
pub mod log_softmax;
pub mod identity;
pub mod cmp_ops;
pub mod math_ops;
pub mod concat;
pub mod tile;
pub mod binary_ops;
pub mod softmax;
pub mod sigmoid;
pub mod elu;
pub mod relu;
pub mod slice;
pub mod sigmoid_cross_entropy;
pub mod softmax_cross_entropy;
pub mod sparse_softmax_cross_entropy;
pub mod gather;
pub mod matmul;
pub mod swap_axes;
pub mod reshape;
pub mod reduction_ops;
pub mod squeeze;
pub mod expand_dims;


/// Represents a operation node in a computation graph.
/// `Tensor` wraps trait-object of this.
pub trait Op {
    /// Name of this op
    fn name(&self) -> &str;

    /// Actually runs this op.
    /// num of inputs : N,
    /// num of outputs: 1
    fn compute(&mut self, xs: &[&NdArray], train: bool) -> NdArray;

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


// Helper function to generate a symbolic tensor
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

/// Creates a placeholder tensor.
///
/// The placeholder tensor is a dynamic input node to the computation graph,
/// which can be filled on evaluation time.
/// To fill the placeholder, use `autograd::Feed`.
///
/// ```
/// extern crate ndarray;
/// use autograd as ag;
///
/// let ref x = ag::placeholder();
/// let ref y = 3 * x;
///
/// // Fills placeholder `x`.
/// let feed_dict = ag::Feed::new().add(x, ndarray::arr1(&[2.]));
/// assert_eq!(6., y.eval_with_input(feed_dict)[0]);
/// ```
#[inline]
pub fn placeholder() -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Placeholder".to_string() }),
        inputs: vec![],
        param: None,
        rank: 0,
    })))
}


/// Creates a shared variable tensor from rust-ndarray's array object.
///
/// The shared variable behaves like any other tensors, except that
/// it can be optimized with gradient descent methods
/// implemented in `autograd::sgd::optimizers`.
/// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples
#[inline]
pub fn variable<T: ndarray::Dimension>(array: ndarray::Array<f32, T>) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Variable".to_string() }),
        inputs: vec![],
        param: Some(array.into_dyn()),
        rank: 0,
    })))
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
/// Adds all input tensors
///
/// All the input tensors must have same shapes
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
/// * `initial_grad` - This is required **if objective is not a scalar**. In most cases,
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
/// take logarithm of it.
pub fn log_softmax(x: &Tensor, axis: isize) -> Tensor
{
    // TODO: Composing from "node level" LogSumExp.
    let op = log_softmax::LogSoftmax { axis: axis };
    apply_op(op, &[x])
}


#[inline]
/// Take softmax along specified axis
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
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor
{
    apply_op(matmul::MatMul, &[a, b])
}


#[inline]
/// Slices input tensor with indices.
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
/// Concatenates (stacks) input tensors along specified axis.
pub fn concat(tensors: &[&Tensor], axis: usize) -> Tensor
{
    apply_op(concat::Concat { axis: axis }, tensors)
}


#[inline]
/// Gathers slices.
///
/// Along `axis`, slices subviews from `param` with `indices`, and then gathers those.
/// For example, this can be used for embedding vector lookup.
///
/// See also https://www.tensorflow.org/api_docs/python/tf/gather.
///
/// # Arguments
/// * `param` - Target of slicing.
/// * `indices` - Index tensor with which slices `param`. This cab be arbitrary shape.
/// * `axis` - Slices sub tensors along this axis.
///
/// # Returns
/// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
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


/// Returns a constant tensor
pub fn zeros(shape: &[usize]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Zeros".to_string() }),
        inputs: vec![],
        param: Some(::ndarray_ext::zeros(shape)),
        rank: 0,
    })))
}


/// Returns a constant tensor
pub fn ones(shape: &[usize]) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Ones".to_string() }),
        inputs: vec![],
        param: Some(::ndarray_ext::ones(shape)),
        rank: 0,
    })))
}


/// Creates a constant tensor.
#[inline]
pub fn constant<T: ndarray::Dimension>(array: ndarray::Array<f32, T>) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Constant".to_string() }),
        inputs: vec![],
        param: Some(array.into_dyn()),
        rank: 0,
    })))
}


/// Creates a constant tensor.
#[inline]
pub fn scalar(a: f32) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Scalar".to_string() }),
        inputs: vec![],
        param: Some(NdArray::from_elem(ndarray::IxDyn(&[1]), a)),
        rank: 0,
    })))
}


/// Creates a constant tensor.
#[inline]
pub fn range(start: usize, end: usize, step: usize) -> Tensor
{
    Tensor(Rc::new(RefCell::new(RawTensor {
        op: Box::new(dummy_op::DummyOp { name: "Scalar".to_string() }),
        inputs: vec![],
        param: Some(
            ndarray::Array1::range(start as f32, end as f32, step as f32).into_dyn(),
        ),
        rank: 0,
    })))
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


/// Outputs values sampled from the log normal distribution.
pub fn log_normal(shape: &[usize], mean: f64, stddev: f64) -> Tensor
{
    let op = random_ops::LogNormal {
        shape: shape.to_vec(),
        mean: mean,
        stddev: stddev,
    };
    apply_op(op, &[])
}
