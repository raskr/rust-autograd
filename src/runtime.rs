use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{self, ComputeContext, OpInput};
use crate::tensor::{Tensor, TensorInternal};
use crate::FxHashMap;
use crate::{Float, Graph};
use std::cell::UnsafeCell;
use std::mem;
use std::sync::{RwLockReadGuard, RwLockWriteGuard};

/// Helper structure for batched evaluation.
///
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
pub struct Eval<'v, 'f, 's, F: Float> {
    scope: &'s Graph<F>,
    buf: Vec<Tensor<'s, F>>,
    feeds: Option<&'f [crate::runtime::Feed<'v, F>]>,
}

impl<'f, 't, 'v, 's, F: Float> Eval<'v, 'f, 's, F> {
    #[inline]
    /// Instantiates a new evaluation session.
    pub fn new(scope: &'s Graph<F>) -> Self {
        Eval {
            feeds: None,
            scope,
            buf: Vec::new(),
        }
    }

    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
    where
        A: AsRef<Tensor<'s, F>>,
    {
        self.buf.push(*x.as_ref());
        self
    }

    /// `feeds` is a sequence of `(placeholder-tensor, its value)`
    pub fn feed(&mut self, feeds: &'f [crate::Feed<'v, F>]) -> &mut Self {
        self.feeds = Some(feeds);
        self
    }

    #[inline]
    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'t [A]) -> &mut Self
    where
        A: AsRef<Tensor<'s, F>>,
    {
        self.buf.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'t self) -> Vec<Result<NdArray<F>, crate::EvalError>> {
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

//#[derive(Debug)]
struct NodeInfo {
    // node: &'t TensorInternal<T>,
    // the len matches the number of outputs of this node
    value_info_list: op::InputArray<ValueInfo>,
}

// #[derive(Debug)]
enum ValueType {
    Owned,
    View,
    Empty,
    ComputeFailed(op::OpError),
}

//#[derive(Debug)]
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
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
}

struct OutputStorageInner<'view, F: Float> {
    owned: Vec<Option<NdArray<F>>>,
    borrowed: Vec<NdArrayView<'view, F>>,
}

impl<'tensor, 'view, 'lock, F: Float> OutputStorage<'view, F> {
    #[inline]
    fn new() -> Self {
        OutputStorage {
            inner: UnsafeCell::new(OutputStorageInner {
                owned: Vec::new(),
                borrowed: Vec::new(),
            }),
        }
    }

    #[inline]
    fn owned_mut(&self) -> &mut Vec<Option<NdArray<F>>> {
        unsafe { &mut (&mut *self.inner.get()).owned }
    }

    #[inline]
    fn owned(&self) -> &[Option<NdArray<F>>] {
        unsafe { &(&*self.inner.get()).owned }
    }

    #[inline]
    fn view_mut(&self) -> &mut Vec<NdArrayView<'view, F>> {
        unsafe { &mut (&mut *self.inner.get()).borrowed }
    }

    #[inline]
    fn view(&self) -> &[NdArrayView<'view, F>] {
        unsafe { &(&*self.inner.get()).borrowed }
    }
}

fn validate_feed_shapes<F: Float>(feeds: &[Feed<F>], g: &Graph<F>) {
    for feed in feeds {
        let shape = feed.value.shape();
        g.access_node(feed.placeholder_id)
            .validate_feed_shape(shape);
    }
}

#[inline]
fn retrieve_feed<'t, 'feeds, 'feed, F: Float>(
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

// Extract output arrays from `ys` and stores into `storage` (and `node`).
fn install_compute_results<'t, 'view, F: Float>(
    results: crate::op::Results<'view, F>,
    storage: &OutputStorage<'view, F>,
    // node: &'t TensorInternal<F>,
) -> NodeInfo {
    let mut value_info_list = op::OutputArray::new();
    for y in results {
        let info = match y {
            Some(Ok(crate::ArrRepr::Owned(val))) => {
                storage.owned_mut().push(Some(val));
                ValueInfo::new(ValueType::Owned, storage.owned().len() - 1)
            }
            Some(Ok(crate::ArrRepr::View(val))) => {
                storage.view_mut().push(val);
                ValueInfo::new(ValueType::View, storage.view().len() - 1)
            }
            Some(Err(e)) => {
                ValueInfo::new(ValueType::ComputeFailed(e), /*dummy = */ 0)
            }
            None => ValueInfo::new(ValueType::Empty, /*dummy = */ 0),
        };
        value_info_list.push(info);
    }
    NodeInfo {
        // node,
        value_info_list,
    }
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
        validate_feed_shapes(feeds, self);

        let mut node_info_map: FxHashMap<usize, NodeInfo> = FxHashMap::default();

        // Storage in which compute results are stored. Accessed through UnsafeCell.
        let storage = OutputStorage::new();

        let mut dfs_stack = Vec::<(&TensorInternal<F>, bool)>::with_capacity(100);
        for t in tensors.iter() {
            dfs_stack.push((t.as_ref().inner(), false));
        }

        while let Some((node, is_parent)) = dfs_stack.pop() {
            if is_parent {
                if Self::would_not_visit(node, &node_info_map) {
                    continue;
                }

                // Aggregate inputs for `in_node`
                let mut xs = Vec::with_capacity(node.in_edges.len());
                let (mut write_guards, mut read_guards) = (Vec::new(), Vec::new());
                for (in_node, &in_idx) in node.in_edges.iter().zip(&node.input_indices) {
                    let input_inner = in_node.get_inner(self);
                    let x = if input_inner.is_placeholder {
                        OpInput::new(retrieve_feed(feeds, in_node.id))
                    } else if let Some(ref lock) = input_inner.variable_array {
                        unsafe {
                            if in_node.mut_usage {
                                write_guards.push(lock.write().unwrap());
                                let inserted = write_guards.len() - 1;
                                OpInput::new_mut(
                                    (*(&mut write_guards[inserted]
                                        as *mut RwLockWriteGuard<NdArray<F>>))
                                        .view_mut(),
                                )
                            } else {
                                read_guards.push(lock.read().unwrap());
                                let inserted = read_guards.len() - 1;
                                OpInput::new(
                                    (*(&mut read_guards[inserted]
                                        as *mut RwLockReadGuard<NdArray<F>>))
                                        .view(),
                                )
                            }
                        }
                    } else if let Some(ref arr) = input_inner.get_constant_array_inner() {
                        OpInput::new(arr.view())
                    } else {
                        // Search the output of other nodes
                        let vi_list = &node_info_map.get(&in_node.id).unwrap().value_info_list;
                        match vi_list.get(in_idx) {
                            Some(vi) => match &vi.ty {
                                ValueType::Owned => {
                                    OpInput::new(storage.owned()[vi.key].as_ref().unwrap().view())
                                }
                                ValueType::View => OpInput::new(storage.view()[vi.key].clone()),
                                ValueType::Empty => {
                                    panic!(
                                        "Attempting to use {}'s output which is empty.",
                                        input_inner.op.name()
                                    );
                                }
                                ValueType::ComputeFailed(ref e) => {
                                    panic!(
                                        "Attempting to use {}'s output which was errored: {}",
                                        input_inner.op.name(),
                                        e.to_string()
                                    );
                                }
                            },
                            None => {
                                panic!(
                                    "Attempting to use {}'s output which was errored",
                                    input_inner.op.name(),
                                );
                            }
                        }
                    };
                    xs.push(x);
                }

                // run compute
                let mut ctx = ComputeContext::new(node, xs);
                node.op.compute(&mut ctx);
                let ys = ctx.extract_outputs();
                if ys.is_empty() {
                    panic!("Bad op implementation: empty return value");
                }
                // register compute result
                // let node_info = install_compute_results(ys, &storage, node); // mut storage
                let node_info = install_compute_results(ys, &storage); // mut storage
                node_info_map.insert(node.id(), node_info);
            } else {
                // Update dfs stack
                dfs_stack.push((node, true));
                // Push children if needed
                for child in &node.in_edges {
                    let child = child.get(self).scoped_inner();
                    if !Self::would_not_visit(child, &node_info_map) {
                        dfs_stack.push((child, false));
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
                let info = &node_info_map.get(&t.id()).unwrap().value_info_list[0];
                match &info.ty {
                    ValueType::Owned => {
                        Ok(mem::replace(&mut storage.owned_mut()[info.key], None).unwrap())
                    }
                    ValueType::View => Ok(storage.view()[info.key].to_owned()),
                    ValueType::Empty => Err(crate::EvalError::Empty),
                    ValueType::ComputeFailed(e) => Err(crate::EvalError::OpError(e.clone())),
                }
            };
            ret.push(arr);
        }
        ret
    }

    #[inline]
    fn would_not_visit<'t>(
        node: &TensorInternal<F>,
        info_map: &FxHashMap<usize, NodeInfo>,
    ) -> bool {
        node.is_placeholder || node.has_persistent_array || info_map.contains_key(&node.id())
    }
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
