//! Defining things related to evaluation of `Tensor`s
use crate::ndarray::ArrayView;
use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op::{self, ComputeContext, InputArray, OpInput};
use crate::smallvec::SmallVec;
use crate::tensor::{Tensor, TensorInternal};
use crate::variable::VariableID;
use crate::{Context, FxHashMap, VariableEnvironment};
use crate::{Float, Graph};
use ndarray::ArcArray;
use std::cell::{Ref, RefMut, UnsafeCell};

/// Unique id for a placeholder tensor
#[derive(Clone, Copy)]
pub enum PlaceholderKey {
    Name(&'static str),
    ID(usize),
}

/// placeholder name or `Tensor` itself.
pub trait Placeholder {
    fn key(&self) -> PlaceholderKey;
}

impl<'g, F: Float> Placeholder for Tensor<'g, F> {
    fn key(&self) -> PlaceholderKey {
        PlaceholderKey::ID(self.id)
    }
}

impl Placeholder for &'static str {
    fn key(&self) -> PlaceholderKey {
        PlaceholderKey::Name(self)
    }
}

/// Helper structure for tensor evaluations.
///
/// `Evaluator` can buffer evaluation targets with useful `push` and `extend` functions
/// and runs batched evaluation.
/// You can also use `feed` method to feed NdArrays to placeholders.
///
/// ```
/// use autograd as ag;
///
/// ag::run(|ctx| {
///    let a = ctx.placeholder("a", &[]);
///    let x = a + a;
///    let y = a * a;
///    let z = a / a;
///
///    let xyz = ctx.evaluator()  // Create a new Evaluator
///        .push(&x)
///        .extend(&[y, z])
///        .feed(a, ag::ndarray::arr0(2.).view()) // Filling the placeholder `a`
///        .run();  // Do eval
///    });
/// ```
pub struct Evaluator<'view, 'graph, 'e, 'n, 'c, F: Float> {
    ctx: &'c Context<'e, 'n, F>,
    buf: Vec<Tensor<'graph, F>>,
    feeder: Feeder<'view, F>,
}

/// Utility for feeding NdArrays to graphs at run-time
///
/// Helpful when used with [optimizers::Optimizer::update](crate::optimizers::Optimizer::update).
/// See the example: [crate::optimizers::momentum_sgd::MomentumSGD]
#[derive(Clone)]
pub struct Feeder<'view, F: Float> {
    feeds: Vec<crate::evaluation::Feed<'view, F>>,
}

impl<'view, F: Float> Feeder<'view, F> {
    #[inline]
    pub fn new() -> Self {
        Self { feeds: Vec::new() }
    }

    /// Pushes ArrayView in this feeder
    #[inline]
    pub fn push<D>(&mut self, key: impl Placeholder, value: ArrayView<'view, F, D>) -> &mut Self
    where
        D: ndarray::Dimension,
        F: Float,
    {
        self.feeds.push(Feed {
            placeholder_key: key.key(),
            value: value.into_dyn(),
        });
        self
    }
}

impl<'g, 'env, 'name, 'view, F: Float> Context<'env, 'name, F> {
    /// Creates a new evaluator
    #[inline]
    pub fn evaluator<'c>(&'c self) -> Evaluator<'view, 'g, 'env, 'name, 'c, F> {
        Evaluator {
            feeder: Feeder { feeds: Vec::new() },
            ctx: self,
            buf: Vec::new(),
        }
    }
}

impl<'tensor, 'view, 'graph, 'e, 'n, 'c, F: Float> Evaluator<'view, 'graph, 'e, 'n, 'c, F> {
    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.buf.push(*x.as_ref());
        self
    }

    /// Pushes a feed to this evaluator.
    pub fn feed<D>(&mut self, key: impl Placeholder, value: ArrayView<'view, F, D>) -> &mut Self
    where
        D: ndarray::Dimension,
    {
        self.feeder.push(key, value.into_dyn());
        self
    }

    /// Sets a pre-instantiated feeder object
    #[inline]
    pub fn set_feeder(&mut self, feeder: Feeder<'view, F>) -> &mut Self {
        self.feeder = feeder;
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
        self.ctx
            .inner
            .eval(self.buf.as_slice(), &self.feeder.feeds, self.ctx.env_handle)
    }
}

#[derive(Clone)]
pub(crate) struct Feed<'v, T: Float> {
    /// The id of the placeholder tensor
    placeholder_key: PlaceholderKey,
    /// A run-time value of the placeholder
    value: NdArrayView<'v, T>,
}

#[derive(Copy, Clone)]
enum ValueType {
    Owned,
    View,
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
    // - items in the storage never gone while evaluation loop (NdArray's relocation is shallow copy).
    inner: UnsafeCell<OutputStorageInner<'view, F>>,
}

struct OutputStorageInner<'view, F: Float> {
    // Each of NdArray is Some right up until eval's ret-val extraction phase.
    // In that phase, each of entry is replaced with None to avoid copying the entire vector.
    value_storage: Vec<Option<NdArray<F>>>,
    view_storage: Vec<NdArrayView<'view, F>>,
}

impl<'view, F: Float> OutputStorage<'view, F> {
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
    fn get(&'view self, vi: ValueInfo) -> NdArrayView<'view, F> {
        match vi.ty {
            ValueType::Owned => self.get_from_owned(vi.key),
            ValueType::View => self.get_from_view(vi.key),
        }
    }
}

// search the feed using `in_node_id`
fn retrieve_feed<'feeds, 'feed, F: Float>(
    feeds: &'feeds [Feed<'feed, F>],
    in_node: &Tensor<F>,
    feed_name: &str,
) -> NdArrayView<'feeds, F> {
    let in_node_id = in_node.id;
    // linear search is tolerable for feeds in most cases.
    for feed in feeds {
        match feed.placeholder_key {
            PlaceholderKey::ID(id) => {
                if in_node_id == id {
                    let ret = feed.value.view();
                    in_node.validate_using_known_shape(ret.shape());
                    return ret;
                }
            }
            PlaceholderKey::Name(name) => {
                if feed_name == name {
                    let ret = feed.value.view();
                    in_node.validate_using_known_shape(ret.shape());
                    return ret;
                }
            }
        }
    }
    panic!("Placeholder unfilled");
}

// Extract output arrays from `results` and stores into `storage`.
fn install_compute_results<'view, F: Float>(
    ys: Result<op::OutputArray<op::OpOutput<'view, F>>, op::OpError>,
    storage: &OutputStorage<'view, F>,
) -> Result<op::OutputArray<ValueInfo>, op::OpError> {
    let mut value_info_list = op::OutputArray::new();
    match ys {
        Ok(ys) => {
            debug_assert!(!ys.is_empty(), "Bad op implementation: empty return value");
            for y in ys {
                match y {
                    crate::OpOutput::Reuse(existing_key) => {
                        // push nothing to storage
                        value_info_list.push(ValueInfo::new(ValueType::Owned, existing_key))
                    }
                    crate::OpOutput::Owned(val) => {
                        let new_key = storage.push_owned(val);
                        value_info_list.push(ValueInfo::new(ValueType::Owned, new_key));
                    }
                    crate::OpOutput::View(val) => {
                        let new_key = storage.push_view(val);
                        value_info_list.push(ValueInfo::new(ValueType::View, new_key));
                    }
                }
            }
            Ok(value_info_list)
        }
        Err(e) => Err(e),
    }
}

struct VariableGuardRegister<'v, F: Float> {
    immutable: Vec<Option<UnsafeCell<Ref<'v, NdArray<F>>>>>,
    mutable: Vec<Option<UnsafeCell<RefMut<'v, NdArray<F>>>>>,
}

impl<'v, 'e, F: Float> VariableGuardRegister<'v, F> {
    fn new(max_size: usize) -> Self {
        let mut immutable = Vec::with_capacity(max_size);
        let mut mutable = Vec::with_capacity(max_size);
        // init with None
        for _ in 0..max_size {
            immutable.push(None);
            mutable.push(None);
        }
        Self { immutable, mutable }
    }

    fn set(&mut self, vid: VariableID, mut_usage: bool, env: &'v VariableEnvironment<'e, F>) {
        if mut_usage {
            debug_assert!(
                self.mutable[vid.0].is_none(),
                "Bad op impl: taking a variable"
            );
            self.mutable[vid.0] = Some(UnsafeCell::new(env.array_list[vid.0].borrow_mut()));
        } else {
            debug_assert!(self.immutable[vid.0].is_none(), "Bad op impl");
            self.immutable[vid.0] = Some(UnsafeCell::new(env.array_list[vid.0].borrow()));
        }
    }

    fn borrow(&self, vid: VariableID, mut_usage: bool) -> OpInput<'v, F> {
        unsafe {
            if mut_usage {
                OpInput::new_mut(
                    (*self.mutable[vid.0]
                        .as_ref()
                        .expect("Variable array is not set")
                        .get())
                    .view_mut(),
                    None,
                )
            } else {
                OpInput::new(
                    (*self.immutable[vid.0]
                        .as_ref()
                        .expect("`Variable array is not set")
                        .get())
                    .view(),
                    None,
                )
            }
        }
    }

    fn unset(&mut self, vid: VariableID, mut_usage: bool) {
        if mut_usage {
            self.mutable[vid.0] = None;
        } else {
            self.immutable[vid.0] = None;
        }
    }
}

impl<F: Float> Graph<F> {
    pub(crate) fn eval<'feed, 'tensor, 'g, A>(
        &'g self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
        ctx: &VariableEnvironment<F>,
    ) -> Vec<Result<NdArray<F>, crate::EvalError>>
    where
        A: AsRef<Tensor<'g, F>> + Copy,
    {
        let mut node_info_map: FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>> =
            FxHashMap::default();

        // Storage in which compute results are stored. Accessed through UnsafeCell.
        let storage = OutputStorage::new();

        let mut variable_guard_register = VariableGuardRegister::new(ctx.array_list.len());

        // Vec<(node_id, is_parent)>
        let mut dfs_stack = Vec::<(usize, bool)>::with_capacity(1 << 10);

        for t in tensors.iter() {
            crate::graph::assert_same_graph(self, t.as_ref().graph);
            dfs_stack.push((t.as_ref().id(), false));
        }

        while let Some((node_id, is_parent)) = dfs_stack.pop() {
            //  in this block, relocation of Graph::node_set's contents must not be occurred
            let node = self.access_inner(node_id);
            if is_parent {
                if would_not_visit(&node, &node_info_map) {
                    continue;
                }

                // =====================================================================================
                // Aggregate input values for `node`. if any of the inputs failed, it's a total failure.
                // =====================================================================================

                let mut xs = InputArray::new();

                let mut input_status = Ok(());

                // Save var guards
                for (in_node, _) in node.in_nodes.iter().zip(&node.input_indices) {
                    if let Some(vid) = in_node.variable_id(self) {
                        // is variable array
                        variable_guard_register.set(vid, in_node.mut_usage, ctx);
                    }
                }

                for (input, &in_idx) in node.in_nodes.iter().zip(&node.input_indices) {
                    // `in_idx` is not 0 only when `in_node` is multi-output op and `node` selects nth value from it using `Graph::nth_tensor`.
                    let in_node = input.as_tensor(self);
                    let x = {
                        if let Some(p_name) = in_node.placeholder_name() {
                            Ok(OpInput::new(retrieve_feed(feeds, &in_node, p_name), None))
                        } else if let Some(vid) = input.variable_id(self) {
                            // is variable array
                            Ok(variable_guard_register.borrow(vid, input.mut_usage))
                        } else {
                            // Search the value of input nodes.
                            match &node_info_map.get(&input.id).unwrap() {
                                Err(e) => Err(e.clone()),
                                Ok(vi_list) => {
                                    let value_info = vi_list[in_idx];
                                    Ok(OpInput::new(storage.get(value_info), Some(value_info.key)))
                                }
                            }
                        }
                    };
                    match x {
                        Ok(x) => xs.push(x),
                        Err(e) => {
                            input_status = Err(e);
                            break;
                        }
                    }
                }

                // ====================================================
                // Run Op::compute() if `node`'s inputs were not failed
                // ====================================================

                let installed_node_info = input_status.and_then(|()| {
                    let mut ctx = ComputeContext::new(xs);
                    let status = node.get_op().compute(&mut ctx);
                    let ret = status.map(|()| ctx.ys);
                    // register compute result
                    install_compute_results(ret, &storage)
                });

                // Release var guards
                for (in_node, _) in node.in_nodes.iter().zip(&node.input_indices) {
                    if let Some(vid) = in_node.variable_id(self) {
                        // is variable array
                        variable_guard_register.unset(vid, in_node.mut_usage);
                    }
                }

                // Cache the result
                node_info_map.insert(node_id, installed_node_info);
            } else {
                // Update dfs stack
                dfs_stack.push((node_id, true));
                // Push children if needed
                for child in &node.in_nodes {
                    let child = self.access_inner(child.id);
                    if !would_not_visit(&child, &node_info_map) {
                        dfs_stack.push((child.id, false));
                    }
                }
            }
        }

        // Aggregate return values
        let mut ret = Vec::with_capacity(tensors.len());
        for t in tensors {
            let t = t.as_ref();
            let arr = if let Some(vid) = t.get_variable_id() {
                // case 1: variable tensor
                Ok(ctx.array_list[vid.0].clone().into_inner())
            } else if let Some(name) = t.placeholder_name() {
                // case 2: placeholder tensor
                Ok(retrieve_feed(feeds, t, name).to_owned())
            } else {
                // case 3: normal tensor
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
    node: &Ref<TensorInternal<F>>,
    info_map: &FxHashMap<usize, Result<op::OutputArray<ValueInfo>, op::OpError>>,
) -> bool {
    node.placeholder_name.is_some() || node.is_variable() || info_map.contains_key(&node.id())
}

#[test]
fn test_eval2() {
    use crate::tensor_ops as T;

    let mut ctx = crate::VariableEnvironment::new();
    ctx.run(|g: &mut Context<f32>| {
        let a = T::ones(&[1, 1], g);
        let b = T::sigmoid(a);
        b.eval(g).unwrap();
    })
}

#[test]
fn test_eval() {
    use crate::tensor_ops as T;

    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let v: Tensor<f32> = g.placeholder("v", &[3, 2, 1]);
        let z = T::reduce_sum(T::squeeze(v, &[2]), &[0, 1], false);
        let grad = T::grad(&[z], &[v]);
        let mut eval = g.evaluator();
        let result = eval
            .extend(&grad)
            .feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())
            .run();
        assert_eq!(result[0].as_ref().unwrap().shape(), &[3, 2, 1]);
    })
}

#[test]
fn test_variable_eval() {
    use crate::variable::GetVariableTensor;
    let mut ctx = VariableEnvironment::new();
    let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let arr_clone = ndarray::arr1(&[0., 0., 0.]).into_dyn();
    let a = ctx.slot().set(arr);
    ctx.run(|g| {
        let av = g.variable(a);
        assert_eq!(Ok(arr_clone), av.eval(g));
    });
}

#[test]
fn test_constant_eval() {
    use crate::tensor_ops::*;
    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Ok(arr.clone()), convert_to_tensor(arr, g).eval(g));
    });
}

#[test]
fn test_placeholder_eval() {
    use crate::tensor_ops::*;
    use ndarray::ShapeBuilder; // Needed for .strides() method

    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let arr: NdArray<f32> = crate::ndarray_ext::ones(&[3, 2, 1]);
        let v = g.placeholder("v", &[3, 2, 1]);
        let mut eval = g.evaluator();
        let result = eval.feed(v, arr.view()).push(v).run();
        assert_eq!(Ok(arr), result[0]);
    });
}

#[test]
fn test_eval3() {
    use crate::tensor_ops::*;

    let mut ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let v: Tensor<f32> = g.placeholder("v", &[3, 2, 1]);
        let v2: Tensor<f32> = g.placeholder("v2", &[3, 2, 1]);
        let b = v + v2;
        let results = g
            .evaluator()
            .push(b)
            .feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())
            .feed("v2", crate::ndarray_ext::ones(&[3, 2, 1]).view())
            .run();
    })
}
