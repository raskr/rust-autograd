//! Defining things related to `ag::op::Op`.
//!
use crate::ndarray_ext::{NdArrayView, NdArrayViewMut};
use crate::smallvec::SmallVec;
use crate::tensor::{Tensor, TensorInternal};
use crate::{Float, NdArray};
use std::any::type_name;
use std::fmt;
use std::marker::PhantomData;
use std::mem;

// Properties for op's `compute` method.
// Actual number of inout/output nodes are around 1~2 in most cases.
pub(crate) const NUM_MAX_OUTPUT: usize = 2;
pub(crate) const NUM_MAX_INPUT: usize = 4;

pub(crate) type InputArray<T> = SmallVec<[T; NUM_MAX_INPUT]>;
pub(crate) type OutputArray<T> = SmallVec<[T; NUM_MAX_OUTPUT]>;

pub(crate) type ComputeResult<'v, T> = Result<crate::ArrRepr<'v, T>, OpError>;

/// Error in `Op`'s computation.
#[derive(Clone, Debug, PartialEq)]
pub enum OpError {
    NdArrayError(String, ndarray::ShapeError),
    IncompatibleShape(String),
    TypeUnsupported(String),
    InvalidDims(String),
    OutOfBounds(String),
}

impl std::error::Error for OpError {}

impl fmt::Display for OpError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            OpError::NdArrayError(pref, e) => write!(f, "{}: ", pref).and_then(|()| e.fmt(f)),
            OpError::IncompatibleShape(s) => write!(f, "{}: ", s),
            OpError::TypeUnsupported(s) => write!(f, "{}: ", s),
            OpError::InvalidDims(s) => write!(f, "{}: ", s),
            OpError::OutOfBounds(s) => write!(f, "{}: ", s),
        }
    }
}

/// Sequence of an op's compute result.
///
/// Op can have multiple output NdArrays.
pub(crate) type Results<'v, T> = OutputArray<Option<ComputeResult<'v, T>>>;

/// Operation trait. `Tensor` wraps trait-object of this.
///
/// # Implementing differentiable operations
///
/// Many of well-known ops are pre-defined in `Graph`, but you can also
/// implement custom ops by hand.
///
/// ```
/// use ndarray;
/// use autograd as ag;
/// use ag::Graph;
///
/// type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
///     fn compute(
///         &self,
///         ctx: &mut ag::op::ComputeContext<T>,
///     ) {
///         let x: &ag::NdArrayView<_> = &ctx.input(0);
///         // Use `ndarray::Array::mapv` for element-wise computation.
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
///         ctx.append_output(y);
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
///         // Symbolic gradient of the input of Sigmoid
///         let gy = ctx.output_grad();
///         let y = ctx.output();
///         let gx = gy * (y - ctx.graph().square(y));
///         ctx.append_input_grad(Some(gx));
///     }
/// }
///
///  use ag::tensor::Input;
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid<'graph, F: ag::Float>(x: &ag::Tensor<'graph, F>, g: &'graph ag::Graph<F>)
/// -> ag::Tensor<'graph, F> {
///     ag::Tensor::builder()
///            .set_inputs(&[Input::new(x)])
///            .build(g, Sigmoid)
/// }
/// ```
pub trait Op<F: Float> {
    /// Name of this op
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    /// Runs this op with `ComputeContext`.
    fn compute(&self, ctx: &mut ComputeContext<F>);

    /// Returns symbolic gradients for input nodes by use of output's gradients etc.
    fn grad(&self, ctx: &mut GradientContext<F>);
}

pub(crate) struct DummyOp<F: Float> {
    pub phantom: PhantomData<F>,
}

impl<F: Float> DummyOp<F> {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        DummyOp {
            phantom: PhantomData,
        }
    }
}

impl<F: Float> Op<F> for DummyOp<F> {
    fn compute(&self, _: &mut ComputeContext<F>) {}
    fn grad(&self, _: &mut GradientContext<F>) {}
}

/// Wrapper object of NdArrayView/NdArrayViewMut which is fed to `Op::compute`
///
/// Used in `OpComputeContext`.
pub(crate) enum OpInput<'v, T: Float> {
    /// Read-only view
    RO(Option<NdArrayView<'v, T>>),
    /// Read-write view
    RW(Option<NdArrayViewMut<'v, T>>),
}

impl<'v, T: Float> OpInput<'v, T> {
    #[inline]
    /// Make a read-only input array
    pub fn new(x: NdArrayView<'v, T>) -> Self {
        OpInput::RO(Some(x))
    }

    #[inline]
    /// Make a read/write input array
    pub fn new_mut(x: NdArrayViewMut<'v, T>) -> Self {
        OpInput::RW(Some(x))
    }
}

/// Context of an `Op`'s computation phase.
///
/// # Example
///
/// ```
/// use autograd as ag;
///
/// // Implementing `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
///     fn compute(
///         &self,
///         ctx: &mut ag::op::ComputeContext<T>,
///     ) {
///         // Getting the first input array.
///         let x: &ag::NdArrayView<_> = &ctx.input(0);
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
///         // Put the computed result.
///         ctx.append_output(y);
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) { /* ... */ }
/// }
/// ```
pub struct ComputeContext<'t, 'v, T: Float> {
    node: &'t TensorInternal<T>,
    // Input arrays
    xs: InputArray<OpInput<'v, T>>,
    // Output arrays
    ys: Results<'v, T>,
}

impl<'g, 't, 'v, T: Float> ComputeContext<'t, 'v, T> {
    pub(crate) fn extract_outputs(self) -> Results<'v, T> {
        self.ys
    }

    #[inline]
    pub(crate) fn new(node: &'t TensorInternal<T>, xs: InputArray<OpInput<'v, T>>) -> Self {
        ComputeContext {
            node,
            xs,
            ys: OutputArray::new(),
        }
    }

    /// Grabs the `i` th input array as a *read-only* array view.
    ///
    /// Calling `input(i)` more than once causes panic.
    #[inline]
    pub fn input(&mut self, i: usize) -> NdArrayView<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!("Bad op impl: input index out of range."),
        };
        match x {
            OpInput::RO(ref mut a) => match a.take() {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: input({})/input_mut({}) cannot be called twice",
                    self.node.get_op().name(),
                    i,
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl of {}: cannot perform immutable borrowing for input({})",
                    self.node.get_op().name(),
                    i
                );
            }
        }
    }

    /// Grabs the `i` th input array as a *read-write* array view.
    ///
    /// Calling `input_mut(i)` more than once causes panic.
    #[inline]
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'v, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!("Bad op impl: {}'s input doesn't exist.", i),
        };
        match x {
            OpInput::RW(ref mut a) => match a.take() {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: input({})/input_mut({}) cannot be called twice",
                    self.node.get_op().name(),
                    i,
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl of {}: cannot perform mutable borrowing for input({})",
                    self.node.get_op().name(),
                    i
                );
            }
        }
    }

    /// Appends an `ndarray::ArrayView` to the back of the output list of the current op.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub fn append_output_view(&mut self, y: NdArrayView<'v, T>) {
        self.ys.push(Some(Ok(crate::ArrRepr::View(y))));
    }

    /// Appends an ndarray to the back of the output list of the current op.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub fn append_output(&mut self, y: NdArray<T>) {
        self.ys.push(Some(Ok(crate::ArrRepr::Owned(y))));
    }

    /// Appends an empty result to the back of the output list of the current op.
    ///
    /// For example, this is used for gradient-descent-optimizers.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub fn append_empty_output(&mut self) {
        self.ys.push(None);
    }

    /// Marks this op's compute session as failed.
    ///
    /// NOTE: After called this function, Op::compute should return early.
    #[inline]
    pub fn set_error(&mut self, y: OpError) {
        self.ys.push(Some(Err(y)));
    }

    /// Returns a number of input arrays.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }
}

/// Context of an `Op`'s gradient propagation phase.
///
/// This is passed to an `Op` through `Op::grad`.
/// `Op::grad` should provide the gradients of its inputs by calling `GradientContext::set_input_grads`.
///
/// Use `graph()` to access `Graph` object for tensor computations.
///
/// ```
/// use autograd as ag;
///
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
///     fn compute(&self, ctx: &mut ag::op::ComputeContext<T>) { /* ... */ }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
///         // Symbolic gradient of the input of Sigmoid
///         let gy = ctx.output_grad();
///         // Symbolic output tensor
///         let y = ctx.output();
///         // `Tensor` computations
///         let gx = gy * (y - ctx.graph().square(y));
///         // Propagates input's gradient.
///         ctx.append_input_grad(Some(gx));
///     }
/// }
/// ```
pub struct GradientContext<'g, T: Float> {
    gy: Tensor<'g, T>,
    y: Tensor<'g, T>,
    graph: &'g crate::graph::Graph<T>,
    gxs: InputArray<Option<Tensor<'g, T>>>,
}

impl<'g, T: Float> GradientContext<'g, T> {
    pub(crate) fn new(
        gy: Tensor<'g, T>,
        y: Tensor<'g, T>,
        graph: &'g crate::graph::Graph<T>,
    ) -> Self {
        GradientContext {
            gy,
            y,
            graph,
            gxs: InputArray::new(),
        }
    }

    // Call Op::grad and return `gxs`
    pub(crate) fn extract_input_grads(mut self) -> InputArray<Option<Tensor<'g, T>>> {
        let id = self.y.id;
        unsafe {
            // steal op
            let stolen = mem::replace(&mut self.graph().access_inner_mut(id).op, None).unwrap();
            // call Op::grad
            stolen.grad(&mut self);
            // restore
            mem::swap(&mut self.graph().access_inner_mut(id).op, &mut Some(stolen));
            debug_assert!(
                !self.gxs.is_empty(),
                "Bad Op impl: GradientContext::set_input_grads was not called"
            );
            self.gxs
        }
    }

    /// Returns the symbolic gradient of the op's output.
    #[inline]
    pub fn output_grad(&self) -> Tensor<'g, T> {
        self.gy
    }

    /// Grabs the symbolic output of the op.
    #[inline]
    pub fn output(&self) -> Tensor<'g, T> {
        self.y
    }

    /// Grabs the `i` th symbolic input.
    #[inline]
    pub fn input(&self, i: usize) -> Tensor<'g, T> {
        return self
            .y
            .input_tensor(i, self.graph)
            .expect("bad Op::grad impl");
    }

    /// Returns the number of inputs.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        unsafe { self.y.inner().in_edges.len() }
    }

    /// Returns a graph object that is usable for tensor computations in the context.
    #[inline]
    pub fn graph(&self) -> &'g crate::graph::Graph<T> {
        self.graph
    }

    /// Back-propagates the input's gradient.
    ///
    /// Appends the given tensor to the back of the input-gradient-list.
    /// `None` argument indicates that the `Op`'s input doesn't have gradient.
    /// Note that `Op::grad` must call this function as many as `num_inputs()`.
    #[inline]
    pub fn append_input_grad(&mut self, gx: Option<Tensor<'g, T>>) {
        self.gxs.push(gx);
    }
}
