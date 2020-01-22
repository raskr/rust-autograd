//! Defining things related to `ag::op::Op`.
//!
use crate::arrayvec::ArrayVec;
use crate::ndarray_ext::{NdArrayView, NdArrayViewMut};
use crate::tensor::{Tensor, TensorInternal};
use crate::Float;
use std::any::type_name;
use std::marker::PhantomData;
use std::mem;

pub(crate) const NUM_MAX_OUTPUT: usize = 8;

/// Sequence of an op's compute result.
///
/// Op can have multiple output arrays.
pub(crate) type Results<'v, T> =
    ArrayVec<[Result<crate::ArrRepr<'v, T>, ComputeException>; NUM_MAX_OUTPUT]>;

#[derive(Clone, Copy, Debug)]
/// Exception in an op's computation.
///
/// Note that this is *exception* more like error.
pub enum ComputeException {
    /// Computation finished correctly with no output (typically used for variable-optimizer's output)
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
/// use ag::Graph;
///
/// type NdArray<T: ag::Float> = ndarray::Array<T, ndarray::IxDyn>;
///
/// // Implements `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::Op<T> for Sigmoid {
///     // In this method, any errors caused by bad user-inputs should results in "panic".
///     // (`ag::op::ComputeException` represents an exception rather than an error.)
///     fn compute(
///         &self,
///         ctx: &mut ag::op::ComputeContext<T>,
///     ) {
///         let x = &ctx.input(0);
///         // Use `ndarray::Array::mapv` for element-wise computation.
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
///         ctx.append_output(Ok(ag::ArrRepr::Owned(y)));
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) {
///         // Symbolic gradient of the input of Sigmoid
///         let gy = ctx.output_grad();
///         let y = ctx.output();
///         let gx = gy * (y - ctx.graph().square(y));
///         ctx.set_input_grads(vec![Some(gx)]);
///     }
/// }
///
/// // Symbolic `sigmoid` function for end-user.
/// fn sigmoid<'graph: 'tensor, 'tensor, F: ag::Float>(x: &ag::Tensor<'tensor, 'graph, F>, g: &'graph ag::Graph<F>)
/// -> ag::Tensor<'tensor, 'graph, F> {
///     ag::Tensor::builder()
///            .set_shape(&g.shape(x))
///            .set_input(x).build(&g, Sigmoid)
/// }
/// ```
pub trait Op<T: Float> {
    /// Name of this op
    fn name(&self) -> &str {
        type_name::<Self>()
    }

    /// Runs this op.
    fn compute(&self, ctx: &mut ComputeContext<T>);

    /// Returns symbolic gradients for input nodes by use of output gradient etc.
    fn grad(&self, ctx: &mut GradientContext<T>);
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

impl<T: Float> Op<T> for DummyOp<T> {
    fn compute(&self, _: &mut ComputeContext<T>) {}
    fn grad(&self, _: &mut GradientContext<T>) {}
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

/// Contains properties used for op's computation.
///
/// # Example
///
/// ```
/// use autograd as ag;
///
/// // Implementing `Op` trait for `Sigmoid`.
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::Op<T> for Sigmoid {
///     // In this method, any errors caused by bad user-inputs should result in "panic".
///     // (`ag::op::ComputeException` represents an exception rather than an error.)
///     fn compute(
///         &self,
///         ctx: &mut ag::op::ComputeContext<T>,
///     ) {
///         // Getting the first input array.
///         let x: &ag::NdArrayView<_> = &ctx.input(0);
///         let half = T::from(0.5).unwrap();
///         let y = x.mapv(move |a| ((a * half).tanh() * half) + half);
///         // Put the computed result.
///         ctx.append_output(Ok(ag::ArrRepr::Owned(y)));
///     }
///
///     fn grad(&self, ctx: &mut ag::op::GradientContext<T>) { /* ... */ }
/// }
/// ```
pub struct ComputeContext<'k, 'v, T: Float> {
    // A `Tensor` object can be looked up in `Op::compute`
    node: &'k TensorInternal<T>,
    // Input arrays
    xs: Vec<OpInput<'v, T>>,
    // Output arrays
    ys: ArrayVec<[Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>; NUM_MAX_OUTPUT]>,
}

impl<'s, 'k: 's, 'v: 's, T: Float> ComputeContext<'k, 'v, T> {
    pub(crate) fn extract_outputs(
        self,
    ) -> ArrayVec<[Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>; NUM_MAX_OUTPUT]>
    {
        self.ys
    }

    #[inline]
    pub(crate) fn new(node: &'k TensorInternal<T>, xs: Vec<OpInput<'v, T>>) -> Self {
        ComputeContext {
            node,
            xs,
            ys: ArrayVec::<[_; NUM_MAX_OUTPUT]>::new(),
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
            OpInput::RO(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: input({})/input_mut({}) cannot be called twice",
                    self.node.op.name(),
                    i,
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl of {}: cannot perform immutable borrowing for input({})",
                    self.node.op.name(),
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
            OpInput::RW(ref mut a) => match mem::replace(a, None) {
                Some(ret) => ret,
                None => panic!(
                    "Bad op impl of {}: input({})/input_mut({}) cannot be called twice",
                    self.node.op.name(),
                    i,
                    i
                ),
            },
            _ => {
                panic!(
                    "Bad op impl of {}: cannot perform mutable borrowing for input({})",
                    self.node.op.name(),
                    i
                );
            }
        }
    }

    #[inline]
    #[allow(dead_code)]
    #[doc(hidden)]
    pub fn set_output(
        &mut self,
        ys: ArrayVec<[Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>; NUM_MAX_OUTPUT]>,
    ) {
        self.ys = ys;
    }

    /// Appends an ndarray to the back of the output list of the current op.
    ///
    /// Implementors of `Op::compute` must not forget to call this method in `Op::compute`, otherwise panic occurs.
    pub fn append_output(&mut self, y: Result<crate::ArrRepr<'v, T>, crate::op::ComputeException>) {
        match self.ys.try_push(y) {
            Ok(()) => {}
            _ => {
                panic!(
                    "Bad op impl: reached the maximum number of output arrays ({})",
                    NUM_MAX_OUTPUT
                );
            }
        }
    }

    /// Returns a number of input arrays.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }
}

/// Contains properties of this op used for gradient propagation.
///
/// This is passed to an `Op` through `Op::grad`.
/// `Op::grad` should provide the gradients of its inputs by calling `GradientContext::set_input_grads`.
///
/// ```
/// use autograd as ag;
///
/// struct Sigmoid;
///
/// impl<T: ag::Float> ag::Op<T> for Sigmoid {
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
///         ctx.set_input_grads(vec![Some(gx)]);
///     }
/// }
/// ```
pub struct GradientContext<'tensor, 'scope: 'tensor, T: Float> {
    gy: Tensor<'tensor, 'scope, T>,
    xs: Vec<Tensor<'tensor, 'scope, T>>,
    y: Tensor<'tensor, 'scope, T>,
    graph: &'scope crate::graph::Graph<T>,
    gxs: Option<Vec<Option<Tensor<'tensor, 'scope, T>>>>,
}

impl<'tensor, 'scope: 'tensor, T: Float> GradientContext<'tensor, 'scope, T> {
    pub(crate) fn new(
        gy: Tensor<'tensor, 'scope, T>,
        xs: Vec<Tensor<'tensor, 'scope, T>>,
        y: Tensor<'tensor, 'scope, T>,
        graph: &'scope crate::graph::Graph<T>,
    ) -> Self {
        GradientContext {
            gy,
            xs,
            y,
            graph,
            gxs: None,
        }
    }

    pub(crate) fn extract_input_grads(self) -> Vec<Option<Tensor<'tensor, 'scope, T>>> {
        self.gxs
            .expect("Bad Op impl: GradientContext::set_input_grads was not called")
    }

    /// Returns the symbolic gradient of the op's output.
    #[inline]
    pub fn output_grad(&self) -> Tensor<'tensor, 'scope, T> {
        self.gy
    }

    /// Grabs the symbolic output of the op.
    #[inline]
    pub fn output(&self) -> Tensor<'tensor, 'scope, T> {
        self.y
    }

    /// Grabs the `i` th symbolic input.
    #[inline]
    pub fn input(&self, i: usize) -> Tensor<'tensor, 'scope, T> {
        self.xs[i]
    }

    /// Returns the number of inputs.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        self.xs.len()
    }

    /// Returns a graph object that is usable for tensor computations
    #[inline]
    pub fn graph(&self) -> &'scope crate::graph::Graph<T> {
        self.graph
    }

    /// Sets gradients of `xs`.
    ///
    /// All the `Op::grad` implementation must call this function even if it is not differentiable.
    #[inline]
    pub fn set_input_grads(&mut self, gxs: Vec<Option<Tensor<'tensor, 'scope, T>>>) {
        self.gxs = Some(gxs);
    }
}
