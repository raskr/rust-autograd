use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{ComputeContext, OpInput};
use crate::tensor::{PersistentArray, Tensor, TensorInternal};
use crate::{hashbrown::hash_map::Entry, FxHashMap, FxHashSet};
use crate::{Float, Graph};
use crossbeam::crossbeam_channel;
use std::cell::UnsafeCell;
use std::mem;
use std::ops::Deref;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard, TryLockError};

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
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
pub struct Eval<'v, 'f, 't, 's, F: Float> {
    scope: &'s Graph<F>,
    buf: Vec<Tensor<'t, 's, F>>,
    feeds: Option<&'f [crate::runtime::Feed<'v, F>]>,
}

impl<'f, 't, 'v, 's: 't, F: Float> Eval<'v, 'f, 't, 's, F> {
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
        A: AsRef<Tensor<'t, 's, F>>,
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
        A: AsRef<Tensor<'t, 's, F>>,
    {
        self.buf.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'t self) -> Vec<Option<NdArray<F>>> {
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

#[derive(Debug)]
struct NodeInfo<'t, T: Float> {
    node: &'t TensorInternal<T>,
    // the len matches the number of outputs of this node
    value_info_list: Vec<ValueInfo>,
}

#[derive(Debug)]
struct NodeWithStateAsync<'t, F: Float> {
    node: &'t TensorInternal<F>,
    successors: Vec<&'t TensorInternal<F>>,
    in_persistent_arrays: Vec<PersistentArray<'t, F>>,
    // idx to lookup evaluation stats.
    target_idx: Option<usize>,
    state: NodeStateAsync,
}

#[derive(Debug)]
struct NodeStateAsync {
    // value_info_list.len() matches the number of outputs of this node.
    value_info_list: Vec<ValueInfo>,
    // initialized with the in-degree of `node`;
    // when this is reduced to 0, `node` is ready to be evaluated.
    pending_count: usize,
    scheduled: bool,
}

impl<'lock, 't, T: Float> Deref for NodeWithStateAsync<'t, T> {
    type Target = TensorInternal<T>;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.node
    }
}

impl<'t, 's, F: Float> NodeWithStateAsync<'t, F> {
    fn new(
        node: &'t TensorInternal<F>,
        successor: Option<&'t TensorInternal<F>>,
        g: &'s Graph<F>,
        target_idx: Option<usize>,
    ) -> Self {
        let mut successors = Vec::new();
        if let Some(suc) = successor {
            if !contains(successors.as_slice(), suc) {
                successors.push(suc);
            }
        }

        // Collect input arrays using graph beforehand since `Graph` can't shared between threads.
        let mut persistent_input_arrays = Vec::with_capacity(node.in_edges.len());
        for x in &node.in_edges {
            persistent_input_arrays.push(x.get_inner(g).get_persistent_array());
        }
        let state = NodeStateAsync {
            pending_count: 0,
            scheduled: false,
            value_info_list: Vec::new(),
        };
        NodeWithStateAsync {
            node,
            successors,
            target_idx,
            state,
            in_persistent_arrays: persistent_input_arrays,
        }
    }

    #[inline]
    fn scheduled(&self) -> bool {
        self.state.scheduled
    }

    #[inline]
    fn mark_scheduled(&mut self) {
        self.state.scheduled = true;
    }

    #[inline]
    fn increment_pending_count(&mut self) {
        self.state.pending_count += 1;
    }

    #[inline]
    fn decrement_pending_count(&mut self) {
        self.state.pending_count = self.state.pending_count.saturating_sub(1);
    }

    #[inline]
    fn ready(&self) -> bool {
        self.state.pending_count == 0
    }
}

// Builds a subgraph consisting of nodes that are reachable from `tensors`.
fn build_stateful_subgraph_from<'t, 's, F, A>(
    targets: &'t [A],
    graph: &'s Graph<F>,
) -> (
    FxHashMap<usize, NodeWithStateAsync<'t, F>>,
    FxHashSet<&'t TensorInternal<F>>,
)
where
    F: Float,
    A: AsRef<Tensor<'t, 's, F>> + Copy,
{
    let mut map = FxHashMap::<usize, NodeWithStateAsync<F>>::default();
    let mut sources = FxHashSet::default();
    let mut dfs_stack: Vec<&TensorInternal<_>> = Vec::with_capacity(128);

    // Initialize the graph and stack with `targets`
    for (i, t) in targets.iter().enumerate() {
        let t = t.as_ref();
        let node = NodeWithStateAsync::new(t.tensor, None, graph, Some(i));
        if let Entry::Vacant(ent) = map.entry(t.id()) {
            let inserted = ent.insert(node);
            dfs_stack.push(inserted.node);
        } else {
            panic!("Detected a duplication in the given evaluation target list.");
        }
    }

    while let Some(node) = dfs_stack.pop() {
        if node.is_source() {
            sources.insert(node);
        }
        for child in &node.in_edges {
            let mut found_new_successor = true;
            match map.entry(child.get(graph).id()) {
                Entry::Vacant(ent) => {
                    // initial visit
                    let inserted = ent.insert(NodeWithStateAsync::new(
                        child.get_inner(graph),
                        Some(node),
                        graph,
                        None,
                    ));
                    dfs_stack.push(inserted.node);
                }
                Entry::Occupied(mut ent) => {
                    let successors = &mut ent.get_mut().successors;
                    // ensuring no duplication in successors to handle the case like `y = add(x, x)`.
                    if !contains(successors.as_slice(), node) {
                        successors.push(node);
                    } else {
                        found_new_successor = false;
                    }
                }
            }
            if found_new_successor {
                map.get_mut(&node.id()).unwrap().increment_pending_count();
            }
        }
    }
    (map, sources)
}

#[inline]
fn contains<T: PartialEq>(slice: &[T], item: T) -> bool {
    for x in slice {
        if *x == item {
            return true;
        }
    }
    false
}

// type of ndarray
#[derive(Clone, Copy, PartialEq, Debug)]
enum ValueType {
    Owned,
    View,
    Empty,
}

#[derive(Clone, Debug)]
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

struct OpEvalResult<'tensor, 'view, T: Float> {
    tensor: &'tensor TensorInternal<T>,
    ys: Option<crate::op::Results<'view, T>>,
    rescheduled: bool,
}

struct ViewStorage<'view, F: Float> {
    inner: Arc<RwLock<Vec<NdArrayView<'view, F>>>>,
}

struct ValueStorage<F: Float> {
    inner: Arc<RwLock<Vec<Option<NdArray<F>>>>>,
}

impl<'view, F: Float> ViewStorage<'view, F> {
    #[inline]
    fn new() -> Self {
        let inner = Arc::new(RwLock::new(Vec::new()));
        Self { inner }
    }

    #[inline]
    fn view(&self, key: usize) -> NdArrayView<'view, F> {
        unsafe {
            let inner = self.inner.read().unwrap();
            let ptr: *const NdArrayView<'view, F> = &*&inner[key];
            (*ptr).clone()
        }
    }

    #[inline]
    // Returns the inserted position.
    fn push(&self, view: NdArrayView<'view, F>) -> usize {
        let mut inner = self.inner.write().unwrap();
        inner.push(view);
        inner.len() - 1
    }
}

impl<'view, F: Float> ValueStorage<F> {
    #[inline]
    fn new() -> Self {
        let inner = Arc::new(RwLock::new(Vec::new()));
        Self { inner }
    }

    #[inline]
    fn owned(&self) -> RwLockWriteGuard<Vec<Option<NdArray<F>>>> {
        self.inner.write().unwrap()
    }

    #[inline]
    fn view(&self, key: usize) -> NdArrayView<'view, F> {
        unsafe {
            let inner = self.inner.read().unwrap();
            let ptr: *const NdArray<F> = inner[key].as_ref().unwrap();
            (*ptr).view()
        }
    }

    #[inline]
    // Returns the inserted position.
    fn push(&self, value: NdArray<F>) -> usize {
        let mut inner = self.inner.write().unwrap();
        inner.push(Some(value));
        inner.len() - 1
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

// map key is a tensor-id
struct LockGuardRegistry<'lock, F: Float> {
    read_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockReadGuard<'lock, NdArray<F>>>>>>,
    write_guards: UnsafeCell<FxHashMap<usize, Vec<Option<RwLockWriteGuard<'lock, NdArray<F>>>>>>,
}

impl<'t, 'lock, F: Float> LockGuardRegistry<'lock, F> {
    #[inline]
    fn init_read(&self, key: &'t TensorInternal<F>) {
        unsafe {
            (&mut *self.read_guards.get()).insert(key.id(), crate::none_vec(key.in_edges.len()));
        }
    }

    #[inline]
    fn init_write(&self, key: &'t TensorInternal<F>) {
        unsafe {
            (&mut *self.write_guards.get()).insert(key.id(), crate::none_vec(key.in_edges.len()));
        }
    }

    #[inline]
    fn new() -> Self {
        LockGuardRegistry {
            read_guards: UnsafeCell::new(FxHashMap::default()),
            write_guards: UnsafeCell::new(FxHashMap::default()),
        }
    }

    #[inline]
    fn register_write<'view>(
        &self,
        node_id: usize,
        input_idx: usize,
        g: RwLockWriteGuard<'lock, NdArray<F>>,
    ) -> &mut RwLockWriteGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> =
                (&mut *self.write_guards.get()).get_mut(&node_id).unwrap();
            got[input_idx] = Some(g);
            got[input_idx].as_mut().unwrap()
        }
    }

    #[inline]
    fn register_read<'view>(
        &self,
        node_id: usize,
        input_idx: usize,
        g: RwLockReadGuard<'lock, NdArray<F>>,
    ) -> &RwLockReadGuard<'lock, NdArray<F>> {
        unsafe {
            let got: &mut Vec<Option<_>> =
                (&mut *self.read_guards.get()).get_mut(&node_id).unwrap();
            got[input_idx] = Some(g);
            let ref_: &RwLockReadGuard<'lock, NdArray<F>> = got[input_idx].as_ref().unwrap();
            ref_
        }
    }

    #[inline]
    fn deregister_input_guards_of(&self, ten: &TensorInternal<F>) {
        for (i, input) in ten.in_edges.iter().enumerate() {
            unsafe {
                if input.mut_usage {
                    mem::swap(
                        &mut (&mut *self.write_guards.get()).get_mut(&ten.id()).unwrap()[i],
                        &mut None,
                    );
                } else {
                    mem::swap(
                        &mut (&mut *self.read_guards.get()).get_mut(&ten.id()).unwrap()[i],
                        &mut None,
                    );
                }
            }
        }
    }
}

struct StatefulSubGraph<'t, F: Float> {
    // node id -> NodeState
    map: UnsafeCell<FxHashMap<usize, NodeWithStateAsync<'t, F>>>,
}

impl<'t, F: Float> StatefulSubGraph<'t, F> {
    #[inline]
    fn state(&self, key: &usize) -> *const NodeWithStateAsync<'t, F> {
        unsafe { (&*self.map.get()).get(key).unwrap() }
    }

    #[inline]
    fn state_mut(&self, key: &usize) -> *mut NodeWithStateAsync<'t, F> {
        unsafe { (&mut *self.map.get()).get_mut(key).unwrap() }
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
    node: &'t TensorInternal<F>,
) -> NodeInfo<'t, F> {
    let mut value_info_list = Vec::with_capacity(results.len());
    for y in results {
        let info = match y {
            Ok(crate::ArrRepr::Owned(val)) => {
                storage.owned_mut().push(Some(val));
                ValueInfo::new(ValueType::Owned, storage.owned().len() - 1)
            }
            Ok(crate::ArrRepr::View(val)) => {
                storage.view_mut().push(val);
                ValueInfo::new(ValueType::View, storage.view().len() - 1)
            }
            _ => ValueInfo::new(ValueType::Empty, /*dummy = */ 0),
        };
        value_info_list.push(info);
    }
    NodeInfo {
        node,
        value_info_list,
    }
}

// Extract output arrays from `ys`.
fn install_compute_results_async<'lock, 't, 'view, F: Float>(
    ys: crate::op::Results<'view, F>,
    value_storage: &ValueStorage<F>,
    view_storage: &ViewStorage<'view, F>,
    node_state: &mut NodeWithStateAsync<'t, F>, // mut actually
) {
    let mut info_list = Vec::with_capacity(ys.len());
    for y in ys {
        let info = match y {
            Ok(crate::ArrRepr::Owned(val)) => {
                let key = value_storage.push(val);
                ValueInfo::new(ValueType::Owned, key) // inserted pos
            }
            Ok(crate::ArrRepr::View(val)) => {
                let key = view_storage.push(val);
                ValueInfo::new(ValueType::View, key)
            }
            _ => ValueInfo::new(ValueType::Empty, /*dummy=*/ usize::default()),
        };
        info_list.push(info);
    }
    node_state.state.value_info_list = info_list;
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum NodeStatus {
    Completed,
    NotYet,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum GraphStatus {
    Completed,
    NotYet,
}

struct CompletionStatus {
    // id -> status
    target_statuses: Vec<(usize, NodeStatus)>,
    whole_status: GraphStatus,
    targets_remaining: usize,
}

impl CompletionStatus {
    // updates targets_remaining if necessary and returns the status
    #[inline]
    fn maybe_update_with<F: Float>(&mut self, evaluated: &NodeWithStateAsync<F>) -> GraphStatus {
        if let Some(idx) = evaluated.target_idx {
            // if `evaluated` is the evaluation target..
            let mut slot = &mut self.target_statuses[idx];
            if slot.1 == NodeStatus::NotYet {
                slot.1 = NodeStatus::Completed;
                // saturated subtraction is not need here.
                self.targets_remaining -= 1;
                if self.targets_remaining == 0 {
                    self.whole_status = GraphStatus::Completed;
                }
            }
        }
        self.whole_status
    }
}

impl<F: Float> Graph<F> {
    #[allow(dead_code)]
    // FIXME: too slow, and deadlock in parallel cargo test
    pub fn eval_async<'feed, 'tensor, 'scope, A>(
        &'scope self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
    ) -> Vec<Option<NdArray<F>>>
    where
        A: AsRef<Tensor<'tensor, 'scope, F>> + Copy,
    {
        // Panics if given shapes are invalid
        validate_feed_shapes(feeds, self);

        let owned_storage = &ValueStorage::new();
        let view_storage = &ViewStorage::new();
        let num_targets = tensors.len();
        let mut target_statuses = Vec::with_capacity(num_targets);
        for t in tensors {
            target_statuses.push((t.as_ref().id(), NodeStatus::NotYet));
        }

        let (state_map, sources) = build_stateful_subgraph_from(tensors, self);

        // rayon scope
        // - blocks until all nodes in the subgraph are processed.
        // - generates the return value of this function
        crate::rayon::scope(move |rayon_scope| {
            let mut completion_status = CompletionStatus {
                target_statuses,
                whole_status: GraphStatus::NotYet,
                targets_remaining: num_targets,
            };
            let graph_state = StatefulSubGraph {
                map: UnsafeCell::new(state_map),
            };
            let (tx, rx) = crossbeam_channel::unbounded();
            let guard_registry = LockGuardRegistry::new();

            // schedule source nodes.
            for &src in &sources {
                tx.send(OpEvalResult {
                    rescheduled: false,
                    tensor: src,
                    ys: if !src.requires_compute() {
                        None
                    } else {
                        let mut ctx = ComputeContext::new(src, Vec::new());
                        src.op.compute(&mut ctx);
                        let ys = ctx.extract_outputs();
                        if ys.is_empty() {
                            panic!("Bad op implementation: empty return value");
                        }
                        Some(ys)
                    },
                })
                .unwrap();
            }

            // main loop.
            loop {
                // aggregate and register a compute result.
                let (status, evaluated) = unsafe {
                    let OpEvalResult {
                        tensor,
                        ys,
                        rescheduled,
                    } = rx.recv().unwrap();
                    guard_registry.deregister_input_guards_of(tensor);
                    let state_mut = graph_state.state_mut(&tensor.id());
                    if !rescheduled {
                        if let Some(ys) = ys {
                            install_compute_results_async(
                                ys,
                                owned_storage,
                                view_storage,
                                &mut *state_mut,
                            );
                        }
                    }
                    let imm = &*state_mut;
                    (completion_status.maybe_update_with(imm), imm)
                };

                if status == GraphStatus::Completed {
                    break;
                }

                // try to schedule the successors of the evaluated node.
                for &suc in &evaluated.successors {
                    unsafe {
                        let suc_info = &mut *graph_state.state_mut(&suc.id());
                        // decrement pending count since an input node of `suc` was processed
                        suc_info.decrement_pending_count();

                        if !suc_info.scheduled() && suc_info.ready() {
                            // Try to schedule `suc`.
                            let suc_input_persistent_arrays = &suc_info.in_persistent_arrays;
                            let mut guard_registry_read_init = false;
                            let mut guard_registry_write_init = false;
                            let mut xs = Vec::with_capacity(suc.in_edges.len());

                            // Aggregate in_node's inputs
                            let mut should_reschedule = false;

                            for (i, ((input, &in_idx), in_arr)) in suc
                                .in_edges
                                .iter()
                                .zip(&suc.input_indices)
                                .zip(suc_input_persistent_arrays)
                                .enumerate()
                            {
                                let x = if input.is_placeholder {
                                    OpInput::new(retrieve_feed(feeds, input.id))
                                } else if let PersistentArray::Variable(ref lock) = in_arr {
                                    if input.mut_usage {
                                        if !guard_registry_write_init {
                                            guard_registry.init_write(suc);
                                            guard_registry_write_init = true;
                                        }
                                        match lock.try_write() {
                                            Ok(guard) => OpInput::new_mut(
                                                (*(&mut *guard_registry.register_write(
                                                    suc.id(),
                                                    i,
                                                    guard,
                                                )
                                                    as *mut RwLockWriteGuard<NdArray<F>>))
                                                    .view_mut(),
                                            ),
                                            Err(TryLockError::WouldBlock) => {
                                                should_reschedule = true;
                                                break;
                                            }
                                            Err(TryLockError::Poisoned(_)) => {
                                                panic!("TryLockError::Poisoned");
                                            }
                                        }
                                    } else {
                                        if !guard_registry_read_init {
                                            guard_registry.init_read(suc);
                                            guard_registry_read_init = true;
                                        }
                                        match lock.try_read() {
                                            Ok(guard) => OpInput::new(
                                                (*(&*guard_registry.register_read(
                                                    suc.id(),
                                                    i,
                                                    guard,
                                                )
                                                    as *const RwLockReadGuard<NdArray<F>>))
                                                    .view(),
                                            ),
                                            Err(TryLockError::WouldBlock) => {
                                                should_reschedule = true;
                                                break;
                                            }
                                            Err(TryLockError::Poisoned(_)) => {
                                                panic!("TryLockError::Poisoned");
                                            }
                                        }
                                    }
                                } else if let PersistentArray::Constant(ref arr) = in_arr {
                                    OpInput::new(arr.view())
                                } else {
                                    // Retrieve the output of other nodes
                                    let input_info = &*graph_state.state(&input.id);
                                    let info = &input_info.state.value_info_list[in_idx];
                                    // NOTE: input views are not tracked by borrow checker but it's ok because
                                    // - Only the main thread can mutate the output storage.
                                    // - Every item in the storage is thread safe.
                                    // - Once an item is placed in the storage, that exists there until the storage dropped.
                                    match info.ty {
                                        ValueType::Owned => {
                                            OpInput::new(owned_storage.view(info.key))
                                        }
                                        ValueType::View => {
                                            OpInput::new(view_storage.view(info.key))
                                        }
                                        ValueType::Empty => {
                                            panic!("Attempting to use an empty output as an op's input.");
                                        }
                                    }
                                };
                                xs.push(x);
                            }
                            // unwrapping Result<(), SendError<T>> is ok since the channel outlives this scope.
                            if should_reschedule {
                                // input aggregation cancelled, rescheduling...
                                tx.send(OpEvalResult {
                                    rescheduled: true,
                                    tensor: evaluated.node,
                                    ys: None,
                                })
                                .unwrap();
                            } else {
                                // schedule the task in the global worker pool
                                let tx = tx.clone();
                                suc_info.mark_scheduled();
                                rayon_scope.spawn(move |_| {
                                    // run compute
                                    let mut ctx = ComputeContext::new(suc, xs);
                                    suc.op.compute(&mut ctx);
                                    let ys = ctx.extract_outputs();
                                    if ys.is_empty() {
                                        panic!("Bad op implementation: empty return value");
                                    }
                                    tx.send(OpEvalResult {
                                        rescheduled: false,
                                        tensor: suc,
                                        ys: Some(ys),
                                    })
                                    .unwrap();
                                });
                            }
                        }
                    }
                }
            }

            // aggregate return values
            let target_statuses = &completion_status.target_statuses;
            let mut ret: Vec<Option<NdArray<F>>> = Vec::with_capacity(target_statuses.len());
            for (id, _) in target_statuses {
                unsafe {
                    let node = &*graph_state.state(id);
                    let owned_value = if let Some(per) = node.clone_persistent_array() {
                        Some(per)
                    } else if node.is_placeholder {
                        Some(retrieve_feed(feeds, *id).to_owned())
                    } else {
                        let info = &node.state.value_info_list[0];
                        match info.ty {
                            ValueType::Owned => {
                                mem::replace(&mut owned_storage.owned()[info.key], None)
                            }
                            ValueType::View => Some(view_storage.view(info.key).to_owned()),
                            ValueType::Empty => None,
                        }
                    };
                    ret.push(owned_value);
                }
            }
            ret
        })
    }

    /// Evaluates given symbolic tensors as a list of `ndarray::Array<F, ndarray::IxDyn>`.
    ///
    /// Each return value can be `None`;
    /// for example, evaluation of `gradient_descent_ops::*`
    /// would result in `None`.
    ///
    /// NOTE: All the runtime errors are not reported by return values, but by *panic*.
    ///
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
    ///     assert_eq!(evaluated[0], Some(array![0., 0.].into_dyn()));
    ///     assert_eq!(evaluated[1], Some(array![1., 1.].into_dyn()));
    /// });
    /// ```
    /// See also [Tensor::eval](tensor/struct.Tensor.html#method.eval).
    pub fn eval<'feed, 'tensor, 'scope, A>(
        &'scope self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
    ) -> Vec<Option<NdArray<F>>>
    where
        A: AsRef<Tensor<'tensor, 'scope, F>> + Copy,
    {
        validate_feed_shapes(feeds, self);

        let mut node_info_map: FxHashMap<usize, NodeInfo<'tensor, F>> = FxHashMap::default();

        // Storage in which compute results are stored. Accessed through UnsafeCell.
        let storage = OutputStorage::new();

        let mut dfs_stack = Vec::<(&TensorInternal<F>, bool)>::with_capacity(100);
        for t in tensors.iter() {
            dfs_stack.push((t.as_ref().tensor, false));
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
                    let input_inner = in_node.get(self).tensor;
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
                    } else if let Some(ref arr) = input_inner.get_constant_array() {
                        OpInput::new(arr.view())
                    } else {
                        // Search the output of other nodes
                        let vi = &node_info_map.get(&in_node.id).unwrap().value_info_list[in_idx];
                        match vi.ty {
                            ValueType::Owned => {
                                OpInput::new(storage.owned()[vi.key].as_ref().unwrap().view())
                                // imm
                            }
                            ValueType::View => OpInput::new(storage.view()[vi.key].clone()),
                            ValueType::Empty => {
                                panic!(
                                    "Attempting to use {}'s output which is empty.",
                                    input_inner.op.name()
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
                let node_info = install_compute_results(ys, &storage, node); // mut storage
                node_info_map.insert(node.id(), node_info);
            } else {
                // Update dfs stack
                dfs_stack.push((node, true));
                // Push children if needed
                for child in &node.in_edges {
                    if !Self::would_not_visit(child.get(self).tensor, &node_info_map) {
                        dfs_stack.push((child.get(self).tensor, false));
                    }
                }
            }
        }

        // Aggregate return values
        let mut ret = Vec::with_capacity(tensors.len());
        for t in tensors {
            let t = t.as_ref();
            let arr = if let Some(per) = t.clone_persistent_array() {
                Some(per)
            } else if t.is_placeholder() {
                Some(retrieve_feed(feeds, t.id()).to_owned())
            } else {
                let info = &node_info_map.get(&t.id()).unwrap().value_info_list[0];
                if ValueType::Owned == info.ty {
                    mem::replace(&mut storage.owned_mut()[info.key], None)
                } else if ValueType::View == info.ty {
                    Some(storage.view()[info.key].to_owned())
                } else {
                    None
                }
            };
            ret.push(arr);
        }
        ret
    }

    #[inline]
    fn would_not_visit<'t>(
        node: &TensorInternal<F>,
        info_map: &FxHashMap<usize, NodeInfo<'t, F>>,
    ) -> bool {
        node.is_placeholder || node.has_persistent_array || info_map.contains_key(&node.id())
    }
}

#[test]
fn test_eval2() {
    crate::with(|g: &mut crate::Graph<f32>| {
        let a = g.ones(&[1, 1]);
        let b = g.sigmoid(a);
        b.eval(&[]);
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
        assert_eq!(Some(arr.clone()), g.variable(arr).eval(&[]));
    });
}

#[test]
fn test_constant_eval() {
    use crate::tensor::Constant;
    crate::with(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Some(arr.clone()), g.constant(arr).eval(&[]));
    });
}

#[test]
fn test_placeholder_eval() {
    crate::with(|g| {
        let arr: NdArray<f32> = crate::ndarray_ext::ones(&[3, 2, 1]);
        let v = g.placeholder(&[3, 2, 1]);
        let eval_result = v.eval(&[v.given(arr.view())]);
        assert_eq!(eval_result, Some(arr));
    });
}
