//! # Implementing differentiable operations
//!
//! Many of well-known ops are pre-defined in [crate::tensor_ops], but you can also
//! implement custom ops by hand.
//! See also [crate::tensor::TensorBuilder].
//!
//! ```
//! use ndarray;
//! use autograd as ag;
//! use autograd::op::OpError;
//! use autograd::tensor_ops::*;
//!
//! type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
//!
//! // Implements `Op` trait for `Sigmoid`.
//! struct Sigmoid;
//!
//! impl<T: ag::Float> ag::op::Op<T> for Sigmoid {
//!     fn compute(
//!         &self,
//!         ctx: &mut ag::op::ComputeContext<T>,
//!     ) -> Result<(), OpError> {
//!         let x: &ag::NdArrayView<_> = &ctx.input(0);
//!         // Use `ndarray::Array::mapv` for element-wise computation.
//!         let half = T::from(0.5).unwrap();
//!         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
//!         ctx.append_output(y);
//!         Ok(())
//!     }
//!
//!     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
//!         // gradient of the output of Sigmoid
//!         let gy = ctx.output_grad();
//!         let y = ctx.output();
//!         // gradient of the input of Sigmoid
//!         let gx = gy * (y - square(y));
//!         ctx.append_input_grad(Some(gx));
//!     }
//! }
//!
//! // `sigmoid` function for end-user.
//! fn sigmoid<'graph, F: ag::Float>(x: &ag::Tensor<'graph, F>, g: &'graph ag::Context<F>)
//! -> ag::Tensor<'graph, F> {
//!     ag::Tensor::builder(g)
//!            .append_input(x, false)
//!            .build(Sigmoid)
//! }
//! ```
//!
use std::any::type_name;
use std::fmt;
use std::marker::PhantomData;
use std::mem;

use crate::ndarray_ext::{NdArrayView, NdArrayViewMut, RawNdArrayView};
use crate::smallvec::SmallVec as RawSmallVec;
use crate::tensor::Tensor;
use crate::{Float, NdArray};
use crate::op::OpInput::NonVariable;

pub(crate) const DEFAULT_NUM_EDGES: usize = 2;

pub(crate) type SmallVec<T> = RawSmallVec<[T; DEFAULT_NUM_EDGES]>;

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

/// Trait for tensor operations. `Tensor` structs wrap this.
pub trait Op<F: Float> {
    /// Name of this op
    fn name(&self) -> &'static str {
        type_name::<Self>()
    }

    /// Runs this op with `ComputeContext`.
    fn compute(&self, ctx: &mut ComputeContext<F>) -> Result<(), OpError>;

    /// Returns gradients for input nodes by use of output's gradients etc.
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
    fn compute(&self, _: &mut ComputeContext<F>) -> Result<(), OpError> {
        Ok(())
    }
    fn grad(&self, _: &mut GradientContext<F>) {}
}

/// Wrapper for NdArrayView/NdArrayViewMut which is fed to `Op::compute`
///
/// Used in `Op::ComputeContext`.
pub(crate) enum OpInput<'v, T: Float> {
    NonVariable(Option<NdArrayView<'v, T>>),
    RdOnlyVariable(Option<NdArrayView<'v, T>>),
    RdWrVariable(Option<NdArrayViewMut<'v, T>>),
}

/// `Op::compute`'s output
#[derive(Clone)]
pub(crate) enum OpOutput<T: Float> {
    Owned(NdArray<T>),
    View(RawNdArrayView<T>),
}

impl<'view, T: Float> OpInput<'view, T> {
    #[inline]
    /// Make a read-only input array
    pub fn new_non_variable(x: NdArrayView<'view, T>) -> Self {
        NonVariable(Some(x))
    }

    #[inline]
    /// Make a read-only input array
    pub fn new_rdonly_variable(x: NdArrayView<'view, T>) -> Self {
        OpInput::RdOnlyVariable(Some(x))
    }

    #[inline]
    /// Make a read/write input array
    pub fn new_rdwr_variable(x: NdArrayViewMut<'view, T>) -> Self {
        OpInput::RdWrVariable(Some(x))
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
///     ) -> Result<(), ag::op::OpError> {
///         // Getting the first input array.
///         let x: &ag::NdArrayView<_> = &ctx.input(0);
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
///         // Put the computed result.
///         ctx.append_output(y);
///         Ok(())
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) { /* ... */ }
/// }
/// ```
pub struct ComputeContext<'v, T: Float> {
    // Input arrays
    xs: SmallVec<OpInput<'v, T>>,
    // Output arrays
    pub(crate) ys: SmallVec<OpOutput<T>>,
}

impl<'graph, 'view, T: Float> ComputeContext<'view, T> {
    #[inline]
    pub(crate) fn new(xs: SmallVec<OpInput<'view, T>>) -> Self {
        ComputeContext {
            xs,
            ys: SmallVec::new(),
        }
    }

    /// Grabs the `i` th input array as a *read-only* array view.
    ///
    /// Calling `input(i)` more than once causes panic.
    #[inline]
    pub fn input(&mut self, i: usize) -> NdArrayView<'view, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!("Bad op impl: input index out of range."),
        };
        match x {
            NonVariable(ref mut a) => match a.take() {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl: input({})/input_mut({}) cannot be called twice",
                    i, i
                ),
            },
            OpInput::RdOnlyVariable(ref mut a) => match a.take() {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl: input({})/input_mut({}) cannot be called twice",
                    i, i
                ),
            },
            OpInput::RdWrVariable(_) => {
                panic!(
                    "Bad op impl: cannot perform mutable borrowing for input({}). Use input_mut() instead.",
                    i
                );
            }
        }
    }

    /// Grabs the `i` th input array as a *read-write* array view.
    ///
    /// Calling `input_mut(i)` more than once causes panic.
    #[inline]
    pub fn input_mut(&mut self, i: usize) -> NdArrayViewMut<'view, T> {
        let x = match self.xs.get_mut(i) {
            Some(x) => x,
            None => panic!("Bad op impl: {}'s input doesn't exist.", i),
        };
        match x {
            OpInput::RdWrVariable(ref mut a) => match a.take() {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl: input({})/input_mut({}) cannot be called twice",
                    i, i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl: cannot perform mutable borrowing for input({})",
                    i
                );
            }
        }
    }

    /// Appends an `ndarray::ArrayView` to the back of the output list of the current op.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub fn append_output_view(&mut self, y: NdArrayView<'view, T>) {
        self.append_output_view_raw(y.raw_view());
    }

    /// Appends an `ndarray::ArrayView` to the back of the output list of the current op.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub(crate) fn append_output_view_raw(&mut self, y: RawNdArrayView<T>) {
        let mut contains_variable_input= false;
        for x in &self.xs {
            match x {
                NonVariable(_) => {},
                _ => contains_variable_input = true
            }
        }
        if contains_variable_input {
            // copy it beforehand to avoid use-after-free
            self.ys.push(OpOutput::Owned(unsafe { y.deref_into_view().to_owned() }));
        } else {
            self.ys.push(OpOutput::View(y));
        }
    }

    #[inline]
    pub fn append_empty_output(&mut self) {
        self.ys.push(OpOutput::Owned(NdArray::zeros(
            crate::ndarray::IxDyn(&[]),
        )));
    }

    /// Appends an ndarray to the back of the output list of the current op.
    ///
    /// NOTE: Implementor of `Op::compute` must not forget to call `append_*` as many as the number of its output in `Op::compute`, otherwise panic occurs.
    #[inline]
    pub fn append_output(&mut self, y: NdArray<T>) {
        self.ys.push(OpOutput::Owned(y));
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
/// `Op::grad` should provide the gradients of its inputs by calling `GradientContext::append_input_grad`.
///
/// Use `graph()` to access `Graph` object for tensor computations.
///
/// ```
/// use autograd as ag;
/// use ag::tensor_ops as T;
///
/// struct Sigmoid;
///
/// impl<F: ag::Float> ag::op::Op<F> for Sigmoid {
///     fn compute(&self, ctx: &mut ag::op::ComputeContext<F>) -> Result<(), ag::op::OpError> {
///         /* ... */
///         Ok(())
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<F>) {
///         // gradient of the input of Sigmoid
///         let gy = ctx.output_grad();
///         // output tensor
///         let y = ctx.output();
///         // `Tensor` computations
///         let gx = gy * (y - T::square(y));
///         // Propagates input's gradient.
///         ctx.append_input_grad(Some(gx));
///     }
/// }
/// ```
pub struct GradientContext<'graph, T: Float> {
    gy: Tensor<'graph, T>,
    y: Tensor<'graph, T>,
    graph: &'graph crate::graph::Graph<T>,
    gxs: SmallVec<Option<Tensor<'graph, T>>>,
}

impl<'graph, T: Float> GradientContext<'graph, T> {
    #[inline]
    pub(crate) fn new(
        gy: Tensor<'graph, T>,
        y: Tensor<'graph, T>,
        graph: &'graph crate::graph::Graph<T>,
    ) -> Self {
        GradientContext {
            gy,
            y,
            graph,
            gxs: SmallVec::new(),
        }
    }

    // Call Op::grad and return `gxs`
    pub(crate) fn compute_input_grads(mut self) -> SmallVec<Option<Tensor<'graph, T>>> {
        let id = self.y.id;
        // steal op
        let stolen = self.graph().access_inner_mut(id).op.take().unwrap();

        // call Op::grad
        stolen.grad(&mut self);

        // restore
        mem::swap(&mut self.graph().access_inner_mut(id).op, &mut Some(stolen));
        debug_assert!(
            !self.gxs.is_empty(),
            "Bad Op impl: GradientContext::append_input_grad was not called"
        );
        self.gxs
    }

    /// Returns the gradient of the op's output.
    #[inline]
    pub fn output_grad(&self) -> Tensor<'graph, T> {
        self.gy
    }

    /// Grabs the output of the op.
    #[inline]
    pub fn output(&self) -> Tensor<'graph, T> {
        self.y
    }

    /// Returns input tensors.
    #[inline]
    pub fn inputs(&self) -> SmallVec<Tensor<'graph, T>> {
        let mut ret = SmallVec::new();
        for input in self.y.get_incoming_tensors().iter() {
            ret.push(self.graph.tensor(input.id));
        }
        ret
    }

    /// Grabs the `i` th input tensor.
    #[inline]
    pub fn input(&self, i: usize) -> Tensor<'graph, T> {
        return self
            .y
            .get_incoming_tensor(i, self.graph)
            .expect("bad Op::grad impl");
    }

    /// Returns the number of inputs.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.y.inner().incoming_nodes.len()
    }

    /// Returns a graph object that is usable for tensor computations in the context.
    #[inline]
    pub fn graph(&self) -> &'graph crate::graph::Graph<T> {
        self.graph
    }

    /// Back-propagates the input's gradient.
    ///
    /// Appends the given tensor to the back of the input-gradient-list.
    /// `None` argument indicates that the `Op`'s input doesn't have gradient.
    /// Note that `Op::grad` must call this function as many as `num_inputs()`.
    #[inline]
    pub fn append_input_grad(&mut self, gx: Option<Tensor<'graph, T>>) {
        self.gxs.push(gx);
    }
}
