use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{self, ComputeContext, InputArray, OpInput};
use crate::smallvec::SmallVec;
use crate::tensor::{Tensor, TensorInternal};
use crate::FxHashMap;
use crate::{Float, Graph};
use std::cell::UnsafeCell;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

const NUM_MAX_EVAL_BUF: usize = 8;

type EvalBuf<T> = SmallVec<[T; NUM_MAX_EVAL_BUF]>;

/// Helper structure for batched evaluation.
///
/// `Eval` structure can buffer evaluation targets with useful `push` and `extend` functions
/// and runs batched evaluation.
/// Use this in case [Tensor::eval](tensor/struct.Tensor.html#method.eval)
/// or [Graph::eval](struct.Graph.html#method.eval) doesn't help.
///
/// ```
/// use autograd as ag;
/// use ndarray;
///
/// ag::with(|g| {
///    let a = g.placeholder(&[]);
///    let x = a + a;
///    let y = a * a;
///    let z = a / a;
///
///    ag::Eval::new(g)
///        .push(&x)
///        .extend(&[y, z])
///        .feed(&[a.given(ndarray::arr0(2.).view())])
///        .run();  // Do eval
///    });
/// ```
pub struct Eval<'view, 'feed, 'graph, F: Float> {
    scope: &'graph Graph<F>,
    buf: EvalBuf<Tensor<'graph, F>>,
    feeds: Option<&'feed [crate::runtime::Feed<'view, F>]>,
}

impl<'feed, 'tensor, 'view, 'graph, F: Float> Eval<'view, 'feed, 'graph, F> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new(scope: &'graph Graph<F>) -> Self {
        Eval {
            feeds: None,
            scope,
            buf: EvalBuf::new(),
        }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.buf.push(*x.as_ref());
        self
    }

    /// `feeds` is a sequence of `(placeholder-tensor, its value)`
    pub fn feed(&mut self, feeds: &'feed [crate::Feed<'view, F>]) -> &mut Self {
        self.feeds = Some(feeds);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'tensor [A]) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.buf.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'tensor self) -> Vec<Result<NdArray<F>, crate::EvalError>> {
        self.scope
            .eval(self.buf.as_slice(), self.feeds.unwrap_or(&[]))
    }
}

/// Links a placeholder tensor and its value at run-time.
///
/// Use `Tensor::given` to instanciate, and
/// ensure that this is passed to `ag::Eval`, `ag::eval` or `Tensor::eval`.
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
///     let feed: ag::Feed<_> = x.given(value.view());
///     x.eval(&[feed]);
/// });
/// ```
pub struct Feed<'feed, T: Float> {
    /// The id of the placeholder tensor
    placeholder_id: usize,
    /// A run-time value of the placeholder
    value: NdArrayView<'feed, T>,
}

impl<'feed, F: Float> Feed<'feed, F> {
    #[inline]
    pub(crate) fn new(placeholder_id: usize, value: NdArrayView<'feed, F>) -> Self {
        Feed {
            placeholder_id,
            value,
        }
    }
}
#[derive(Copy, Clone)]
enum ValueType {
    Owned,
    View,
    Empty,
}

#[derive(Copy, Clone)]
struct ValueInfo {
    ty: ValueType,
    // key to lookup output
    key: usize,
}

impl ValueInfo {
    #[inline]
    fn new(ty: ValueType, key: usize) -> Self {
        ValueInfo { ty, key }
    }
}

struct OutputStorage<'view, F: Float> {
    // - storage itself is not shared between threads
    // - items in the storage never gone while evaluation loop.
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
}

struct OutputStorageInner<'view, F: Float> {
    // Each of NdArray is Some right up until eval's ret-val extraction phase.
    // In that phase, each of entry is replaced with None to avoid copying the entire vector.
    value_storage: Vec<Option<NdArray<F>>>,
    view_storage: Vec<NdArrayView<'view, F>>,
}

impl<'tensor, 'view, 'lock, F: Float> OutputStorage<'view, F> {
    #[inline]
    fn new() -> Self {
        OutputStorage {
            inner: UnsafeCell::new(OutputStorageInner {
                value_storage: Vec::new(),
                view_storage: Vec::new(),
            }),
        }
    }

    #[inline]
    unsafe fn inner(&self) -> &OutputStorageInner<'view, F> {
        &*self.inner.get()
    }

    #[inline]
    unsafe fn inner_mut(&self) -> &mut OutputStorageInner<'view, F> {
        &mut *self.inner.get()
    }

    #[inline]
    fn push_owned(&self, val: NdArray<F>) -> usize {
        unsafe {
            let s = &mut self.inner_mut().value_storage;
            let ret = s.len();
            s.push(Some(val));
            ret
        }
    }

    #[inline]
    fn push_view(&self, view: NdArrayView<'view, F>) -> usize {
        unsafe {
            let s = &mut self.inner_mut().view_storage;
            let ret = s.len();
            s.push(view);
            ret
        }
    }

    #[inline]
    fn get_from_view(&self, i: usize) -> NdArrayView<'view, F> {
        unsafe { self.inner().view_storage[i].clone() }
    }

    #[inline]
    fn get_from_owned(&self, i: usize) -> NdArrayView<F> {
        unsafe { self.inner().value_storage[i].as_ref().unwrap().view() }
    }

    #[inline]
    fn take_from_owned(&self, i: usize) -> NdArray<F> {
        unsafe { self.inner_mut().value_storage[i].take().unwrap() }
    }

    #[inline]
    fn get(&'view self, node: &TensorInternal<F>, vi: ValueInfo) -> NdArrayView<'view, F> {
        match vi.ty {
            ValueType::Owned => self.get_from_owned(vi.key),
            ValueType::View => self.get_from_view(vi.key),
            ValueType::Empty => {
                panic!(
                    "Attempting to use {}'s output which is empty.",
                    node.get_op().name()
                );
            }
        }
    }
}

#[inline]
// search the feed using `in_node_id`
fn retrieve_feed<'feeds, 'feed, F: Float>(
    feeds: &'feeds [Feed<'feed, F>],
    in_node_id: usize,
) -> NdArrayView<'feeds, F> {
    // linear search is tolerable for feeds in most cases.
    for feed in feeds {
        if feed.placeholder_id == in_node_id {
            return feed.value.view();
        }
    }
    panic!("Placeholder unfilled");
}

// Extract output arrays from `results` and stores into `storage`.
fn install_compute_results<'view, F: Float>(
    results: crate::op::Results<'view, F>,
    storage: &OutputStorage<'view, F>,
) -> Result<op::OutputArray<ValueInfo>, op::OpError> {
    let mut value_info_list = op::OutputArray::new();
    for y in results {
        match y {
            Some(Ok(crate::ArrRepr::Owned(val))) => {
                let key = storage.push_owned(val);
                value_info_list.push(ValueInfo::new(ValueType::Owned, key));
            }
            Some(Ok(crate::ArrRepr::View(val))) => {
                let key = storage.push_view(val);
                value_info_list.push(ValueInfo::new(ValueType::View, key));
            }
            Some(Err(e)) => {
                return Err(e);
            }
            None => {
                value_info_list.push(ValueInfo::new(ValueType::Empty, /*dummy = */ 0))
            }
        };
    }
    Ok(value_info_list)
}

// aggregated ones are pushed in `input_values`.
// input's status is returned.
#[inline]
fn aggregate_op_inputs<'ret, 'tensor: 'ret, 'slice: 'ret, 'feed: 'slice, F: Float>(
    node: &'tensor TensorInternal<F>,
    g: &Graph<F>,
    node_info_map: &FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>>,
    feeds: &'slice [Feed<'feed, F>],
    storage: &'ret OutputStorage<'ret, F>,
    input_values: &mut InputArray<OpInput<'ret, F>>, // target
    read_guards: &mut InputArray<RwLockReadGuard<'tensor, NdArray<F>>>, // guard storage for variable arrays
    write_guards: &mut InputArray<RwLockWriteGuard<'tensor, NdArray<F>>>, // guard storage for variable arrays
) -> Result<(), op::OpError> {
    let mut input_status = Ok(());

    for (in_node, &in_idx) in node.in_edges.iter().zip(&node.input_indices) {
        // `in_idx` is not 0 only when `in_node` is multi-output op and `node` selects nth value from it using `Graph::nth_tensor`.
        let x = unsafe {
            let input_inner: &TensorInternal<F> = in_node.get_internal(g);
            if input_inner.is_placeholder {
                Ok(OpInput::new(retrieve_feed(feeds, in_node.id)))
            } else if let Some(ref lock) = input_inner.variable_array {
                if in_node.mut_usage {
                    write_guards.push(lock.write().unwrap());
                    let inserted = write_guards.len() - 1;
                    Ok(OpInput::new_mut(
                        (*(&mut write_guards[inserted] as *mut RwLockWriteGuard<NdArray<F>>))
                            .view_mut(),
                    ))
                } else {
                    read_guards.push(lock.read().unwrap());
                    let inserted = read_guards.len() - 1;
                    Ok(OpInput::new(
                        (*(&mut read_guards[inserted] as *mut RwLockReadGuard<NdArray<F>>)).view(),
                    ))
                }
            } else if let Some(arr) = input_inner.get_constant_array_inner() {
                Ok(OpInput::new(arr.view()))
            } else {
                // Search the value of input nodes.
                match &node_info_map.get(&in_node.id).unwrap() {
                    Err(e) => Err(e.clone()),
                    Ok(vi_list) => Ok(OpInput::new(storage.get(input_inner, vi_list[in_idx]))),
                }
            }
        };
        match x {
            Ok(x) => input_values.push(x),
            Err(e) => {
                input_status = Err(e);
                break;
            }
        }
    }
    input_status
}

impl<F: Float> Graph<F> {
    /// Evaluates given symbolic tensors as a list of `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    /// Unlike [Tensor::eval](tensor/struct.Tensor.html#method.eval), this function
    /// supports batched evaluation.
    ///
    /// See also [Eval](struct.Eval.html).
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let a = g.zeros(&[2]);
    ///     let b = g.ones(&[2]);
    ///
    ///     // eval two tensors at once.
    ///     let evaluated = g.eval(&[a, b], &[]);
    ///     assert_eq!(evaluated[0], Ok(array![0., 0.].into_dyn()));
    ///     assert_eq!(evaluated[1], Ok(array![1., 1.].into_dyn()));
    /// });
    /// ```
    pub fn eval<'feed, 'tensor, 'scope, A>(
        &'scope self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
    ) -> Vec<Result<NdArray<F>, crate::EvalError>>
    where
        A: AsRef<Tensor<'scope, F>> + Copy,
    {
        let mut node_info_map: FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>> =
            FxHashMap::default();

        // Storage in which compute results are stored. Accessed through UnsafeCell.
        let storage = OutputStorage::new();

        let mut dfs_stack = Vec::<(&TensorInternal<F>, bool)>::with_capacity(100);
        unsafe {
            for t in tensors.iter() {
                dfs_stack.push((t.as_ref().inner(), false));
            }

            while let Some((node, is_parent)) = dfs_stack.pop() {
                if is_parent {
                    if would_not_visit(node, &node_info_map) {
                        continue;
                    }

                    // Aggregate input values for `node`. if any of the inputs failed, it's a total failure.
                    let mut xs = InputArray::new();
                    let (mut write_guards, mut read_guards) =
                        (InputArray::new(), InputArray::new());
                    let input_status = aggregate_op_inputs(
                        node,
                        self,
                        &node_info_map,
                        feeds,
                        &storage,
                        &mut xs,
                        &mut read_guards,
                        &mut write_guards,
                    );

                    // run compute if `node`'s inputs were not failed
                    let installed_node_info = input_status.and_then(|()| {
                        let mut ctx = ComputeContext::new(node, xs);
                        node.get_op().compute(&mut ctx);
                        let ys = ctx.extract_outputs();
                        debug_assert!(!ys.is_empty(), "Bad op implementation: empty return value");
                        // register compute result
                        install_compute_results(ys, &storage)
                    });
                    node_info_map.insert(node.id(), installed_node_info);
                } else {
                    // Update dfs stack
                    dfs_stack.push((node, true));
                    // Push children if needed
                    for child in &node.in_edges {
                        let child = child.get_internal(self);
                        if !would_not_visit(child, &node_info_map) {
                            dfs_stack.push((child, false));
                        }
                    }
                }
            }
        }

        // Aggregate return values
        let mut ret = Vec::with_capacity(tensors.len());
        for t in tensors {
            let t = t.as_ref();
            let arr = if let Some(per) = t.clone_persistent_array() {
                Ok(per)
            } else if t.is_placeholder() {
                Ok(retrieve_feed(feeds, t.id()).to_owned())
            } else {
                match &node_info_map.get(&t.id()).unwrap() {
                    Ok(value_info_list) => match value_info_list[0] {
                        ValueInfo {
                            ty: ValueType::Owned,
                            key,
                        } => Ok(storage.take_from_owned(key)),
                        ValueInfo {
                            ty: ValueType::View,
                            key,
                        } => Ok(storage.get_from_view(key).to_owned()),
                        ValueInfo {
                            ty: ValueType::Empty,
                            key: _,
                        } => Err(crate::EvalError::Empty),
                    },
                    Err(e) => {
                        // convert to EvalError
                        Err(crate::EvalError::OpError(e.clone()))
                    }
                }
            };
            ret.push(arr);
        }
        ret
    }
}

#[inline]
fn would_not_visit<F: Float>(
    node: &TensorInternal<F>,
    info_map: &FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>>,
) -> bool {
    node.is_placeholder || node.has_persistent_array || info_map.contains_key(&node.id())
}

#[test]
fn test_eval2() {
    crate::with(|g: &mut crate::Graph<f32>| {
        let a = g.ones(&[1, 1]);
        let b = g.sigmoid(a);
        b.eval(&[]).unwrap();
    })
}

#[test]
fn test_eval() {
    crate::with(|g| {
        let v: Tensor<f32> = g.placeholder(&[3, 2, 1]);
        let z = g.reduce_sum(g.squeeze(v, &[2]), &[0, 1], false);
        let g = g.grad(&[z], &[v]);
        let eval_result = g[0].eval(&[v.given(crate::ndarray_ext::ones(&[3, 2, 1]).view())]);
        assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
    })
}

#[test]
fn test_variable_eval() {
    use crate::tensor::Variable;
    crate::with(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Ok(arr.clone()), g.variable(arr).eval(&[]));
    });
}

#[test]
fn test_constant_eval() {
    use crate::tensor::Constant;
    crate::with(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Ok(arr.clone()), g.constant(arr).eval(&[]));
    });
}

#[test]
fn test_placeholder_eval() {
    crate::with(|g| {
        let arr: NdArray<f32> = crate::ndarray_ext::ones(&[3, 2, 1]);
        let v = g.placeholder(&[3, 2, 1]);
        let eval_result = v.eval(&[v.given(arr.view())]);
        assert_eq!(Ok(arr), eval_result);
    });
}
