//! Defining things related to `ag::Tensor`.
use crate::op;
use crate::Float;
use crate::{NdArray, NdArrayView};

use crate::graph::Graph;
use crate::op::GradientContext;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Add, Div, Mul, Sub};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Lazy N-dimensional array.
///
/// `Tensor` is:
///
/// - created by operations of a `Graph`.
/// - not evaluated until `Tensor::eval`, `Graph::eval` or `Eval::run` is called.
/// - cheap to `Copy` since it contains only refs to the owned internal objects.
///
/// The builtin operations for tensors are provided as [Graph's methods](../graph/struct.Graph.html).
///
/// ```
/// use autograd as ag;
///
/// ag::with(|graph| {  // `Graph` is necessary to create tensors.
///     // `random` is just a symbolic object belongs to `graph`.
///     let random: ag::Tensor<f64> = graph.standard_normal(&[2, 3]);
///
///     // This is ok since tensor's binary operators are overloaded!
///     let mul = random * 3.;
///
///     // Evaluates the symbolic tensor as an ndarray::Array<T, IxDyn>.
///     type NdArray = ag::NdArray<f64>;
///     let mul_val: Result<NdArray, ag::EvalError> = mul.eval(&[]);
///
///     // Reshapes the tensor without copy (ArrayView is used internally).
///     let reshaped = graph.reshape(random, &[6]);
///
///     // Evaluating multiple tensors at once.
///     // Note that although `random` node is required two times in this computation graph,
///     // it's evaluated only once since `eval()` is smart enough to avoid duplicated computations.
///     let pair: Vec<Result<NdArray, _>> = graph.eval(&[mul, reshaped], &[]);
/// });
/// ```
#[derive(Clone, Copy)]
pub struct Tensor<'graph, F: Float> {
    pub(crate) id: usize,
    pub(crate) graph: &'graph Graph<F>,
}

impl<'graph, F: Float> Tensor<'graph, F> {
    pub(crate) fn input_tensor(&self, i: usize, g: &'graph Graph<F>) -> Option<Tensor<'graph, F>> {
        unsafe { self.inner().in_edges.get(i).map(|x| x.as_tensor(g)) }
    }

    #[inline]
    pub(crate) unsafe fn inner<'t>(&self) -> &'t TensorInternal<F> {
        self.graph.access_inner(self.id)
    }

    /// Returns the graph to which this tensor belongs.
    #[inline]
    pub fn graph(&self) -> &'graph Graph<F> {
        self.graph
    }

    /// Returns a mutable ref of the graph to which this tensor belongs.
    #[inline]
    pub fn graph_mut(&mut self) -> &'graph Graph<F> {
        &mut self.graph
    }

    /// Evaluates this tensor as an `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.zeros(&[2]);
    ///    assert_eq!(a.eval(&[]), Ok(array![0., 0.].into_dyn()));
    /// });
    /// ```
    ///
    /// See also [Graph::eval](../graph/struct.Graph.html#method.eval).
    pub fn eval<'v>(
        &self,
        feeds: &'v [crate::runtime::Feed<'v, F>],
    ) -> Result<NdArray<F>, crate::EvalError> {
        let mut ret = self.graph.eval(&[self], feeds);
        debug_assert_eq!(ret.len(), 1);
        ret.remove(0)
    }

    /// Retruns a `Feed` assigning a given value to this (placeholder) tensor.
    ///
    /// Ensure that the return value is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let x = g.placeholder(&[2]);
    ///
    ///     // Fills the placeholder with an ArrayView, then eval.
    ///     let value = array![1., 1.];
    ///     x.eval(&[
    ///         x.given(value.view())
    ///     ]);
    /// });
    /// ```
    pub fn given<D>(self, value: ndarray::ArrayView<F, D>) -> crate::runtime::Feed<F>
    where
        D: ndarray::Dimension,
    {
        assert!(
            self.is_placeholder(),
            "Receiver of Tensor::given must be a placeholder."
        );
        unsafe {
            self.inner().validate_feed_shape(value.shape());
        }
        crate::runtime::Feed::new(self.id(), value.into_dyn())
    }

    #[inline]
    /// Creates a new [TensorBuilder](struct.TensorBuilder.html).
    pub fn builder() -> TensorBuilder<F> {
        // Starts with default values
        TensorBuilder {
            shape: None,
            in_edges: op::InputArray::new(),
            can_have_gradient: true,
            constant_array: None,
            variable_array: None,
            is_placeholder: false,
            input_indices: None,
            backprop_inputs: None,
            known_shape: None,
        }
    }

    // Registers a hook on the receiver tensor.
    //
    // ```
    // use autograd as ag;
    //
    // ag::with(|g| {
    //     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).register_hook(ag::hook::Show);
    //     let b: ag::Tensor<f32> = g.ones(&[2, 3]).register_hook(ag::hook::ShowShape);
    //     let c = g.matmul(a, b);
    //
    //     c.eval(&[]);
    //     // [[0.0, 0.0],
    //     // [0.0, 0.0],
    //     // [0.0, 0.0],
    //     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    //
    //     // [2, 3]
    // });
    // ```
    #[inline]
    fn register_hook<H: crate::hook::Hook<F> + 'static>(self, hook: H) -> Tensor<'graph, F> {
        Tensor::builder()
            .append_input(&self)
            .build(self.graph, crate::ops::hook_ops::HookOp::new(hook))
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stderr.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show();
    ///     a.eval(&[]);
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    ///     });
    /// ```
    #[inline]
    pub fn show(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Show)
    }

    /// Sets a hook that displays the evaluation result of the receiver tensor to stderr, with given prefix.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[4, 2]).show_with("My value:");
    ///     a.eval(&[]);
    ///     // My value:
    ///     // [[0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0],
    ///     // [0.0, 0.0]] shape=[4, 2], strides=[2, 1], layout=C (0x1)
    /// });
    ///
    /// ```
    #[inline]
    pub fn show_with(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowWith(what))
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stderr.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape();
    ///     a.eval(&[]);
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape(self) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowShape)
    }

    /// Sets a hook that displays the shape of the evaluated receiver tensor to stderr, with given prefix.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).show_shape_with("My shape:");
    ///     a.eval(&[]);
    ///     // My shape:
    ///     // [2, 3]
    /// });
    /// ```
    #[inline]
    pub fn show_shape_with(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::ShowShapeWith(what))
    }

    /// Sets a hook that displays the given string after evaluation of the receiver tensor.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).print("This is `a`");
    ///     a.eval(&[]);
    ///     // This is `a`
    /// });
    /// ```
    #[inline]
    pub fn print(self, what: &'static str) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Print(what))
    }

    /// Sets a hook that calls the given closure after evaluation of the receiver tensor.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a: ag::Tensor<f32> = g.zeros(&[2, 3]).raw_hook(|arr| println!("{:?}", arr));
    ///     a.eval(&[]);
    /// });
    /// ```
    #[inline]
    pub fn raw_hook<FUN: Fn(&NdArrayView<F>) -> () + 'static + Send + Sync>(
        self,
        f: FUN,
    ) -> Tensor<'graph, F> {
        self.register_hook(crate::hook::Raw {
            raw: f,
            phantom: PhantomData,
        })
    }

    /// Returns the id of this tensor in this graph.
    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_inputs(&self) -> usize {
        unsafe { self.inner().num_inputs() }
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub fn num_backprop_inputs(&self) -> usize {
        unsafe {
            let inner = self.inner();
            inner
                .backprop_inputs
                .as_ref()
                .unwrap_or(&inner.in_edges)
                .len()
        }
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool {
        unsafe { self.inner().is_source() }
    }

    #[inline]
    /// Input nodes used when backprop.
    ///
    pub fn get_backprop_input(&self, idx: usize) -> Input {
        unsafe { self.inner().get_backprop_inputs()[idx].clone() }
    }

    #[inline]
    pub fn is_placeholder(&self) -> bool {
        unsafe { self.inner().is_placeholder }
    }

    #[inline]
    pub fn clone_persistent_array(&self) -> Option<NdArray<F>> {
        unsafe { self.inner().clone_persistent_array() }
    }

    #[inline]
    pub fn get_variable_array(&self) -> Option<&Arc<RwLock<NdArray<F>>>> {
        unsafe {
            // self.inner().variable_array.as_ref().map(|x| x.clone())
            self.inner().variable_array.as_ref()
        }
    }

    #[inline]
    pub fn get_variable_array_ptr(&self) -> Option<*const RwLock<NdArray<F>>> {
        unsafe { self.inner().get_variable_array_inner() }
    }

    #[inline]
    pub fn lock_variable_array(&self) -> Option<RwLockReadGuard<NdArray<F>>> {
        unsafe { self.inner().lock_variable_array() }
    }

    #[inline]
    pub fn lock_variable_array_mut(&self) -> Option<RwLockWriteGuard<NdArray<F>>> {
        unsafe { self.inner().lock_variable_array_mut() }
    }

    #[inline]
    pub fn is_differentiable(&self) -> bool {
        unsafe { self.inner().is_differentiable }
    }

    /// True is this tensor was created by `Graph::variable`.
    #[inline]
    #[allow(dead_code)]
    pub(crate) fn is_variable(&self) -> bool {
        unsafe { self.inner().variable_array.is_some() }
    }
}

impl<'b, T: Float> AsRef<Tensor<'b, T>> for Tensor<'b, T> {
    #[inline(always)]
    fn as_ref(&self) -> &Tensor<'b, T> {
        self
    }
}

pub(crate) struct TensorInternal<F: Float> {
    pub(crate) id: usize,

    // Operation to evaluate this tensor.
    pub(crate) op: Option<Box<dyn op::Op<F>>>,

    // References to immediate predecessors.
    pub(crate) in_edges: op::InputArray<Input>,

    // The rank number for topological ordering in a graph.
    pub(crate) top_rank: usize,

    // *Symbolic* shape of this tensor.
    pub(crate) shape: Option<usize>,

    // An optional *persistent* NdArray.
    //
    // This is `Some` if this tensor is made from `ag::variable`.
    pub(crate) variable_array: Option<Arc<RwLock<NdArray<F>>>>,

    // An optional *persistent* NdArray.
    //
    // This is `Some` if this tensor is made from `ag::constant`.
    pub(crate) constant_array: Option<Arc<NdArray<F>>>,

    // This tensor is placeholder or not.
    pub(crate) is_placeholder: bool,

    // This is true if this tensor can have gradient for any objectives.
    pub(crate) is_differentiable: bool,

    // This is `Some` if this tensor is made from `ag::constant` or `ag::variable`.
    pub(crate) has_persistent_array: bool,

    /// Input indices of arrays used in `compute`
    pub(crate) input_indices: op::InputArray<usize>,

    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) backprop_inputs: Option<op::InputArray<Input>>,

    /// Static shape of this tensor.
    /// Each dim size is *signed* for placeholders.
    pub(crate) known_shape: Option<KnownShape>,
}

impl<F: Float> TensorInternal<F> {
    /// Returns the Op of this tensor
    pub fn get_op(&self) -> &Box<dyn op::Op<F>> {
        self.op
            .as_ref()
            .expect("bad impl: Op is now stolen in gradient.rs")
    }

    #[inline(always)]
    pub fn id(&self) -> usize {
        self.id
    }

    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub(crate) fn is_source(&self) -> bool {
        self.in_edges.is_empty()
    }

    /// Returns the number of inputs of this tensor.
    #[inline]
    pub(crate) fn num_inputs(&self) -> usize {
        self.in_edges.len()
    }

    /// Returns a reference to the persistent constant array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::constant`; otherwise `None`
    #[inline]
    pub(crate) fn get_variable_array_inner(&self) -> Option<*const RwLock<NdArray<F>>> {
        match &self.variable_array {
            Some(ref inner) => Some(&**inner),
            None => None,
        }
    }

    /// Returns a reference to the persistent constant array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::constant`; otherwise `None`
    #[inline]
    pub(crate) fn get_constant_array_inner(&self) -> Option<&NdArray<F>> {
        match &self.constant_array {
            Some(ref inner) => Some(&**inner),
            None => None,
        }
    }

    /// Locks the persistent variable tensor and returns the handle.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::variable`; otherwise `None`.
    #[inline]
    pub(crate) fn lock_variable_array(&self) -> Option<RwLockReadGuard<NdArray<F>>> {
        if let Some(ref arr) = self.variable_array {
            Some(arr.read().unwrap())
        } else {
            None
        }
    }

    /// Returns a mutable reference to the persistent array.
    ///
    /// Note that this is `Some` if this tensor derived from `ag::variable`; otherwise `None`
    #[inline]
    pub(crate) fn lock_variable_array_mut(&self) -> Option<RwLockWriteGuard<NdArray<F>>> {
        if let Some(ref arr) = self.variable_array {
            Some(arr.write().unwrap())
        } else {
            None
        }
    }

    /// Returns a cloned persistent array.
    #[inline]
    pub(crate) fn clone_persistent_array(&self) -> Option<NdArray<F>> {
        if let Some(ref arr) = self.variable_array {
            Some((*arr.read().unwrap()).clone())
        } else if let Some(ref arr) = self.constant_array {
            Some((**arr).clone())
        } else {
            None
        }
    }

    /// True if the op of this tensor is differentiable
    #[inline]
    #[allow(dead_code)]
    pub fn is_differentiable(&self) -> bool {
        self.is_differentiable
    }

    #[inline]
    pub(crate) fn validate_feed_shape(&self, shape: &[usize]) {
        debug_assert!(self.is_placeholder);
        if !self.known_shape.as_ref().unwrap().validate(shape) {
            panic!(
                "Shape error: placeholder required {:?}, but got {:?}",
                self.known_shape.as_ref().unwrap().get(),
                shape
            );
        }
    }

    #[inline]
    /// Input nodes used when backprop.
    ///
    /// This is same as `inputs` in most cases.
    pub(crate) fn get_backprop_inputs(&self) -> &[Input] {
        self.backprop_inputs
            .as_ref()
            .unwrap_or(&self.in_edges)
            .as_slice()
    }
}

impl<T: Float> fmt::Debug for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Node name: {}, id: {}, num of inputs: {}, in-edges: {:?}",
            self.get_op().name(),
            self.id(),
            self.in_edges.len(),
            self.in_edges
        )
    }
}

// empty implementation
impl<T: Float> Eq for TensorInternal<T> {}

impl<T: Float> PartialEq for TensorInternal<T> {
    #[inline(always)]
    fn eq(&self, other: &TensorInternal<T>) -> bool {
        // compare addresses on the heap
        self.id() == other.id()
    }
}

/// Raw pointer hashing
impl<T: Float> Hash for TensorInternal<T> {
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl<T: Float> AsRef<TensorInternal<T>> for TensorInternal<T> {
    #[inline(always)]
    fn as_ref(&self) -> &TensorInternal<T> {
        self
    }
}

impl<T: Float> fmt::Display for TensorInternal<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "name={}", self.get_op().name(),)
    }
}

/// A decorated `Tensor` passed to `TensorBuilder::set_inputs`.
///
/// Use `new` to create an immutable input, or `new_mut` to create a modifiable one.
/// See also [TensorBuilder](struct.TensorBuilder.html).
#[derive(Clone, Debug)]
pub struct Input {
    pub(crate) id: usize,
    pub(crate) mut_usage: bool,
}

impl<'graph> Input {
    /// Instantiates a new immutable `Input` object.
    ///
    /// Run-time value of `val` is passed as an `ndarray::ArrayView` in `Op::compute`.
    #[inline]
    pub fn new<F: Float>(val: &Tensor<'graph, F>) -> Input {
        Input {
            id: val.id(),
            mut_usage: false,
        }
    }

    /// Instantiates a new mutable `Input` object.
    ///
    /// Run-time value of `val` is passed as an `ArrayViewMut` in `Op::compute`.
    #[inline]
    pub fn new_mut<F: Float>(val: &Tensor<'graph, F>) -> Input {
        Input {
            id: val.id(),
            mut_usage: true,
        }
    }

    #[inline]
    pub fn as_tensor<F: Float>(&self, graph: &'graph Graph<F>) -> Tensor<'graph, F> {
        graph.tensor(self.id)
    }

    #[inline]
    pub(crate) unsafe fn get_internal<F: Float>(&self, graph: &Graph<F>) -> &TensorInternal<F> {
        graph.access_inner(self.id)
    }
}

/// Builder for `ag::Tensor` returned by [Tensor::builder](struct.Tensor.html#method.builder).
///
/// This structure is required only when constructing user-defined `Op`.
/// ```
/// use autograd as ag;
/// use ag::tensor::Input;
///
/// struct DummyOp {
///    a: f32
/// }
///
/// impl ag::op::Op<f32> for DummyOp {
///     fn compute(&self, _: &mut ag::op::ComputeContext<f32>) {}
///     fn grad(&self, _: &mut ag::op::GradientContext<f32>) {}
/// }
///
/// ag::with(|g: &mut ag::Graph<f32>| {
///     let input = &g.zeros(&[0]);
///     let my_output: ag::Tensor<_> = ag::Tensor::builder()
///         .set_inputs(&[
///             Input::new(input), // immutable input
///             Input::new_mut(input) // mutable input
///         ])
///         .build(g, DummyOp {a: 42.});
/// });
/// ```
pub struct TensorBuilder<F: Float> {
    shape: Option<usize>,
    in_edges: op::InputArray<Input>,
    can_have_gradient: bool,
    is_placeholder: bool,
    constant_array: Option<Arc<NdArray<F>>>,
    variable_array: Option<Arc<RwLock<NdArray<F>>>>,
    input_indices: Option<op::InputArray<usize>>,
    backprop_inputs: Option<op::InputArray<Input>>,
    known_shape: Option<KnownShape>,
}

pub(crate) struct KnownShape {
    shape: Vec<isize>,
    #[allow(dead_code)]
    is_fully_defined: bool,
}

impl KnownShape {
    pub(crate) fn new(shape: Vec<isize>) -> Self {
        let mut is_fully_defined = true;
        for &a in &shape {
            if a == -1 {
                is_fully_defined = false;
            } else if a <= -1 || a == 0 {
                panic!("Given shape ({:?}) contains invalid dim size(s)", &shape);
            }
        }
        Self {
            shape,
            is_fully_defined,
        }
    }

    #[inline]
    pub fn get(&self) -> &[isize] {
        self.shape.as_slice()
    }

    pub fn validate(&self, target: &[usize]) -> bool {
        if self.shape.len() != target.len() {
            return false;
        }
        for (&i, &u) in self.shape.iter().zip(target) {
            if i > 0 && i as usize != u {
                return false;
            }
        }
        true
    }

    #[inline]
    #[allow(dead_code)]
    pub fn is_fully_defined(&self) -> bool {
        self.is_fully_defined
    }
}

#[test]
fn test_build() {
    crate::with(|s| {
        let a: Tensor<f32> = s.zeros(&[4, 2]);
        let v: Tensor<f32> = s.zeros(&[2, 3]);
        let b: Tensor<f32> = s.zeros(&[4, 3]);
        let z = s.matmul(a, v) + b;
        unsafe {
            let mut vars = [a.inner(), v.inner(), b.inner(), z.inner()];
            // `sort_by_key` don't reverse the order of `a` and `v`
            vars.sort_by_key(|a| a.top_rank);
            assert_eq!(vars, [a.inner(), v.inner(), b.inner(), z.inner()])
        }
    });
}

impl<'graph, F: Float> TensorBuilder<F> {
    #[inline]
    pub(crate) fn set_known_shape(mut self, s: Vec<isize>) -> TensorBuilder<F> {
        self.known_shape = Some(KnownShape::new(s));
        self
    }

    #[inline]
    pub(crate) fn set_shape(mut self, s: &Tensor<'graph, F>) -> TensorBuilder<F> {
        self.shape = Some(s.id());
        self
    }

    #[inline]
    pub fn set_differentiable(mut self, differentiable: bool) -> TensorBuilder<F> {
        self.can_have_gradient = differentiable;
        self
    }

    #[inline]
    /// Sets input tensors.
    /// See also [Input](struct.Input.html).
    pub fn set_inputs(mut self, a: &[Input]) -> TensorBuilder<F> {
        self.in_edges = op::InputArray::from(a);
        self
    }

    #[inline]
    /// Sets input tensors (vector).
    /// See also [Input](struct.Input.html).
    pub fn set_inputs_vec(mut self, a: Vec<Input>) -> TensorBuilder<F> {
        self.in_edges = op::InputArray::from_vec(a);
        self
    }

    #[inline]
    /// Sets read-only input tensors.
    pub(crate) fn set_ro_inputs(mut self, a: &[&Tensor<F>]) -> TensorBuilder<F> {
        for &x in a {
            self.in_edges.push(Input::new(x));
        }
        self
    }

    #[inline]
    pub(crate) fn append_input(mut self, val: &Tensor<F>) -> TensorBuilder<F> {
        self.in_edges.push(Input::new(val));
        self
    }

    #[inline]
    pub(crate) fn set_is_placeholder(mut self, a: bool) -> TensorBuilder<F> {
        self.is_placeholder = a;
        self
    }

    #[inline]
    pub(crate) fn set_constant_array(mut self, a: Arc<NdArray<F>>) -> TensorBuilder<F> {
        self.constant_array = Some(a);
        self
    }

    #[inline]
    pub(crate) fn set_variable_array(mut self, a: Arc<RwLock<NdArray<F>>>) -> TensorBuilder<F> {
        self.variable_array = Some(a);
        self
    }

    #[inline]
    pub(crate) fn set_input_indices(mut self, a: &[usize]) -> TensorBuilder<F> {
        self.input_indices = Some(op::InputArray::from_slice(a));
        self
    }

    #[inline]
    /// Sets inputs for backprop.
    ///
    /// Not required unless backprop-inputs are differs from normal-case inputs
    pub fn set_backprop_inputs(mut self, a: &[Input]) -> TensorBuilder<F> {
        self.backprop_inputs = Some(op::InputArray::from(a));
        self
    }

    /// Finalizes this builder and creates a tensor with given `Op` in the graph.
    pub fn build<O>(self, graph: &'graph Graph<F>, op: O) -> Tensor<'graph, F>
    where
        O: op::Op<F> + 'static,
    {
        let rank = if self.in_edges.is_empty() {
            0
        } else {
            self.in_edges
                .iter()
                .map(|a| unsafe { a.get_internal(graph).top_rank })
                .max()
                .map(|a| a + 1)
                .unwrap_or(0)
        };

        let input_indices = if let Some(a) = self.input_indices {
            assert_eq!(
                a.len(),
                self.in_edges.len(),
                "input_indices.len() must match inputs length"
            );
            a
        } else {
            smallvec::smallvec!(0; self.in_edges.len())
        };

        let new = TensorInternal {
            // `id` is set in `Graph::install`
            id: usize::default(),
            op: Some(Box::new(op)),
            in_edges: self.in_edges,
            top_rank: rank,
            shape: self.shape,
            has_persistent_array: self.variable_array.is_some() || self.constant_array.is_some(),
            variable_array: self.variable_array,
            constant_array: self.constant_array,
            is_placeholder: self.is_placeholder,
            is_differentiable: self.can_have_gradient,
            input_indices,
            backprop_inputs: self.backprop_inputs,
            known_shape: self.known_shape,
        };
        Tensor {
            id: graph.install(new),
            graph,
        }
    }
}

pub(crate) struct Dummy;

impl<T: Float> crate::op::Op<T> for Dummy {
    fn compute(&self, _: &mut crate::op::ComputeContext<T>) {}
    fn grad(&self, _: &mut GradientContext<T>) {}
}

// -- std::ops::{Add, Sub, Mul, Div} implementations --
macro_rules! impl_bin_op_between_tensor_and_float_trait {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Float
        impl<'b, T: Float> $trt<T> for Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.graph.$func(&self, &self.graph.scalar(rhs))
            }
        }

        // &Tensor op Float
        impl<'l, 'b, T: Float> $trt<T> for &'l Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: T) -> Self::Output {
                self.graph.$func(self, &self.graph.scalar(rhs))
            }
        }
    };
}

macro_rules! impl_bin_op_between_tensor_and_primitive {
    ($trt:ident, $func:ident, $op:ident, $scalar_type:ty) => {
        // primitive op Tensor
        impl<'r, 'b, T: Float> $trt<Tensor<'b, T>> for $scalar_type {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: Tensor<'b, T>) -> Self::Output {
                rhs.graph
                    .$func(&rhs.graph.scalar(T::from(self).unwrap()), &rhs)
            }
        }

        // primitive op &Tensor
        impl<'r, 'b, T: Float> $trt<&'r Tensor<'b, T>> for $scalar_type {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: &'r Tensor<'b, T>) -> Self::Output {
                rhs.graph
                    .$func(&rhs.graph.scalar(T::from(self).unwrap()), rhs)
            }
        }
    };
}

impl_bin_op_between_tensor_and_float_trait!(Add, add, AddOp);
impl_bin_op_between_tensor_and_float_trait!(Sub, sub, SubOp);
impl_bin_op_between_tensor_and_float_trait!(Mul, mul, MulOp);
impl_bin_op_between_tensor_and_float_trait!(Div, div, DivOp);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f64);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f64);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f64);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f64);

impl_bin_op_between_tensor_and_primitive!(Add, add, AddOp, f32);
impl_bin_op_between_tensor_and_primitive!(Sub, sub, SubOp, f32);
impl_bin_op_between_tensor_and_primitive!(Mul, mul, MulOp, f32);
impl_bin_op_between_tensor_and_primitive!(Div, div, DivOp, f32);

macro_rules! impl_bin_op_between_tensors {
    ($trt:ident, $func:ident, $op:ident) => {
        // Tensor op Tensor
        impl<'b, T: Float> $trt for Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: Tensor<'b, T>) -> Self::Output {
                self.graph.$func(&self, &rhs)
            }
        }

        // Tensor op &Tensor
        impl<'r, 'b, T: Float> $trt<&'r Tensor<'b, T>> for Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: &'r Tensor<'b, T>) -> Self::Output {
                self.graph.$func(&self, rhs)
            }
        }

        // &Tensor op Tensor
        impl<'l, 'b, T: Float> $trt<Tensor<'b, T>> for &'l Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: Tensor<'b, T>) -> Self::Output {
                self.graph.$func(self, &rhs)
            }
        }

        // &Tensor op &Tensor
        // lifetime of the two tensors are unrelated
        impl<'l, 'r, 'b, T: Float> $trt<&'r Tensor<'b, T>> for &'l Tensor<'b, T> {
            type Output = Tensor<'b, T>;
            fn $func(self, rhs: &'r Tensor<'b, T>) -> Self::Output {
                self.graph.$func(self, rhs)
            }
        }
    };
}

impl_bin_op_between_tensors!(Add, add, AddOp);
impl_bin_op_between_tensors!(Sub, sub, SubOp);
impl_bin_op_between_tensors!(Mul, mul, MulOp);
impl_bin_op_between_tensors!(Div, div, DivOp);

/// Implementors can be converted to `Tensor`.
pub trait AsTensor<'graph, T: Float> {
    fn as_tensor(&self, graph: &'graph Graph<T>) -> Tensor<'graph, T>;
}

impl<'graph, T: Float> AsTensor<'graph, T> for Tensor<'graph, T> {
    fn as_tensor(&self, _: &'graph Graph<T>) -> Tensor<'graph, T> {
        *self
    }
}

macro_rules! impl_as_tensor_for_array {
    ($num_elems:expr) => {
        impl<'graph, T: Float, I: crate::Int> AsTensor<'graph, T> for [I; $num_elems] {
            fn as_tensor(&self, graph: &'graph Graph<T>) -> Tensor<'graph, T> {
                let vec = self
                    .iter()
                    .map(|&a| T::from(a).unwrap())
                    .collect::<Vec<T>>();

                // unwrap is safe
                let arr = NdArray::from_shape_vec(ndarray::IxDyn(&[self.len()]), vec).unwrap();
                graph.convert_to_tensor(arr)
            }
        }
    };
}

impl_as_tensor_for_array!(0);
impl_as_tensor_for_array!(1);
impl_as_tensor_for_array!(2);
impl_as_tensor_for_array!(3);
impl_as_tensor_for_array!(4);
impl_as_tensor_for_array!(5);
impl_as_tensor_for_array!(6);
impl_as_tensor_for_array!(7);
impl_as_tensor_for_array!(8);

/// Trait to create constant tensors.
pub trait Constant<'scope, F: Float, Src: Sized> {
    /// Creates a (persistent) constant tensor from an `NdArray`, or `Arc<NdArray>` to prevent move.
    ///
    /// ```
    /// use std::sync::Arc;
    /// use ndarray::{self, array, IxDyn, Ix1, Array};
    /// use autograd as ag;
    /// // import the trait
    /// use ag::tensor::Constant;
    ///
    /// let v1: Array<f64, Ix1> = array![2.];
    /// let v2: Arc<Array<f64, IxDyn>> = Arc::new(array![2.].into_dyn());
    ///
    /// ag::with(|g| {
    ///    // instantiate from NdArray
    ///    let v1: ag::Tensor<f64> = g.constant(v1);
    ///    // instantiate from `Arc<NdArray>` that allows ref-count.
    ///    let v2: ag::Tensor<f64> = g.constant(v2.clone());
    ///    let y: ag::Tensor<f64> = 3. * v1 * v2;
    ///
    ///    assert_eq!(12., y.eval(&[]).unwrap()[0]);
    /// });
    /// ```
    fn constant(&'scope self, arr: Src) -> Tensor<'scope, F>;
}

/// Trait to create variable tensors.
pub trait Variable<'scope, F: Float, Src: Sized> {
    /// Creates a shared variable tensor from an `NdArray`, or `Arc<RwLock<NdArray>>` to prevent move.
    ///
    /// A shared variable can be mutated with gradient descent methods
    /// implemented in `autograd::gradient_descent_ops`.
    /// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples.
    /// ```
    /// use std::sync::{Arc, RwLock};
    /// use ndarray::{self, array, IxDyn, Ix1, Array};
    /// use autograd as ag;
    /// // import the trait
    /// use ag::tensor::Variable;
    ///
    /// let v1: Array<f64, Ix1> = array![2.];
    /// let v2: Arc<RwLock<Array<f64, IxDyn>>> = ag::ndarray_ext::into_shared(array![2.]);
    ///
    /// ag::with(|g| {
    ///    // Instantiate from an NdArray
    ///    let v1: ag::Tensor<f64> = g.variable(v1);
    ///    // Instantiate from an Arc<RwLock<NdArray>>
    ///    let v2: ag::Tensor<f64> = g.variable(v2);
    ///    let y: ag::Tensor<f64> = 3. * v1 * v2;
    ///
    ///    assert_eq!(12., y.eval(&[]).unwrap()[0]);
    /// });
    /// ```
    fn variable(&'scope self, arr: Src) -> Tensor<'scope, F>;
}

// method overload 1
impl<'graph, F: Float> Constant<'graph, F, Arc<ndarray::Array<F, ndarray::IxDyn>>>
    for crate::graph::Graph<F>
{
    #[inline]
    fn constant(&'graph self, arr: Arc<ndarray::Array<F, ndarray::IxDyn>>) -> Tensor<'graph, F> {
        Tensor::builder()
            .set_constant_array(arr)
            .build(self, crate::ops::basic_source_ops::Const)
    }
}

// method overload 2
macro_rules! impl_constant_dim {
    ($d:ty) => {
        impl<'graph, F: Float> Constant<'graph, F, ndarray::Array<F, $d>>
            for crate::graph::Graph<F>
        {
            #[inline]
            fn constant(&'graph self, arr: ndarray::Array<F, $d>) -> Tensor<'graph, F> {
                Tensor::builder()
                    .set_constant_array(Arc::new(arr.into_dyn()))
                    .build(self, crate::ops::basic_source_ops::Const)
            }
        }
    };
}

// method overload 1
impl<'graph, F: Float> Variable<'graph, F, Arc<RwLock<ndarray::Array<F, ndarray::IxDyn>>>>
    for crate::graph::Graph<F>
{
    #[inline]
    fn variable(
        &'graph self,
        arr: Arc<RwLock<ndarray::Array<F, ndarray::IxDyn>>>,
    ) -> Tensor<'graph, F> {
        Tensor::builder()
            .set_variable_array(arr)
            .build(self, crate::ops::basic_source_ops::Variable)
    }
}

// method overload 2
macro_rules! impl_variable_dim {
    ($d:ty) => {
        impl<'graph, F: Float> Variable<'graph, F, ndarray::Array<F, $d>>
            for crate::graph::Graph<F>
        {
            #[inline]
            fn variable(&'graph self, arr: ndarray::Array<F, $d>) -> Tensor<'graph, F> {
                Tensor::builder()
                    .set_variable_array(Arc::new(RwLock::new(arr.into_dyn())))
                    .build(self, crate::ops::basic_source_ops::Variable)
            }
        }
    };
}

impl_constant_dim!(ndarray::Ix0);
impl_constant_dim!(ndarray::Ix1);
impl_constant_dim!(ndarray::Ix2);
impl_constant_dim!(ndarray::Ix3);
impl_constant_dim!(ndarray::Ix4);
impl_constant_dim!(ndarray::Ix5);
impl_constant_dim!(ndarray::Ix6);
impl_constant_dim!(ndarray::IxDyn);

impl_variable_dim!(ndarray::Ix0);
impl_variable_dim!(ndarray::Ix1);
impl_variable_dim!(ndarray::Ix2);
impl_variable_dim!(ndarray::Ix3);
impl_variable_dim!(ndarray::Ix4);
impl_variable_dim!(ndarray::Ix5);
impl_variable_dim!(ndarray::Ix6);
impl_variable_dim!(ndarray::IxDyn);
