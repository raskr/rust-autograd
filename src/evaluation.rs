use crate::ndarray::ArrayView;
use crate::ndarray_ext::{NdArray, NdArrayView, RawNdArrayView};
use crate::op::{self, OpInput};

use crate::tensor::{Tensor, TensorInternal};
use crate::{Context, FxHashMap, VariableEnvironment};
use crate::{Float, Graph};

use std::cell::Ref;
use crate::graph::TensorID;

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
pub struct Evaluator<'graph, 'env, 'feed, F: Float> {
    graph: &'graph Graph<F>,
    var_env: &'env VariableEnvironment<F>,
    feeder: Feeder<'feed, F>,
    eval_targets: Vec<Tensor<'graph, F>>,
}

/// Utility for feeding NdArrays to graphs at run-time
///
/// Helpful when used with [optimizers::Optimizer::update](crate::optimizers::Optimizer::update).
///
/// ```
/// use autograd as ag;
/// use ag::prelude::*;
/// use ag::optimizers;
/// use ag::optimizers::MomentumSGD;
///
/// type Tensor<'g> = ag::Tensor<'g, f64>;
/// let mut env = ag::VariableEnvironment::new();
/// let opt = MomentumSGD::default("sgd", env.default_namespace().current_var_ids(), &mut env);
///
/// env.run(|g| {
///    let p = g.placeholder("p", &[]);
///
///    let mut feeder = ag::Feeder::new();
///    let feed = ag::ndarray::arr0(2.);
///    feeder.push(p, feed.view());
///
///    let (params, grads): (&[Tensor], &[Tensor]) = (&[], &[]); // dummy here
///    opt.update(params, grads, g, feeder); // do parameter update
/// });
/// ```
#[derive(Clone)]
pub struct Feeder<'view, F: Float> {
    feeds: Vec<Feed<'view, F>>,
}

impl<'view, F: Float> Feeder<'view, F> {
    #[inline]
    pub fn new() -> Self {
        Self { feeds: Vec::new() }
    }

    /// Pushes an `ArrayView` in this feeder
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

impl<'graph, 'env, 'view, F: Float> Context<'env, F> {
    /// Creates a new evaluator
    #[inline]
    pub fn evaluator<'c: 'graph + 'env>(&'c self) -> Evaluator<'graph, 'env, 'view, F> {
        Evaluator {
            feeder: Feeder { feeds: Vec::new() },
            var_env: self.var_env_ref,
            graph: &self.graph,
            eval_targets: Vec::new(),
        }
    }
}

impl<'tensor, 'view, 'graph, 'env, 'ctx, F: Float> Evaluator<'graph, 'env, 'view, F> {
    #[inline]
    /// Appends a tensor to the back of the evaluation targets.
    pub fn push<A>(&mut self, x: A) -> &mut Self
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        self.eval_targets.push(*x.as_ref());
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
        self.eval_targets.extend(xs.iter().map(|x| *x.as_ref()));
        self
    }

    #[inline]
    /// Evaluates the buffered tensors.
    pub fn run(&'tensor self) -> Vec<Result<NdArray<F>, crate::EvalError>> {
        self.graph
            .eval(self.eval_targets.as_slice(), &self.feeder.feeds, self.var_env)
    }
}

#[derive(Clone)]
pub(crate) struct Feed<'view, T: Float> {
    /// The id of the placeholder tensor
    placeholder_key: PlaceholderKey,
    /// A run-time value of the placeholder
    value: NdArrayView<'view, T>,
}

// Storage in which compute results are stored.
struct OpOutputStorage<F: Float> {
    inner: FxHashMap<TensorID, Result<op::SmallVec<OpOutput<F>>, op::OpError>>
}

use crate::op::OpOutput;

impl<F: Float> OpOutputStorage<F> {
    fn new() -> Self {
        OpOutputStorage {
            inner: FxHashMap::default()
        }
    }

    #[inline]
    fn insert(&mut self, key: TensorID, output: Result<op::SmallVec<OpOutput<F>>, op::OpError>) {
        self.inner.insert(key, output);
    }

    #[inline]
    fn get(&self, key: TensorID, selector: usize) -> Result<RawNdArrayView<F>, op::OpError> {
        match self.inner.get(&key).unwrap() {
            Ok(ys) => {
                match &ys[selector] {
                    OpOutput::Owned(arr) => Ok(arr.raw_view()),
                    OpOutput::View(arr) => Ok(arr.clone())
                }
            },
            Err(e) => Err(e.clone())
        }
    }

    fn take(&mut self, key: TensorID, selector: usize) -> Result<NdArray<F>, op::OpError> {
        self.inner.remove(&key).unwrap().and_then(|mut ys| {
            // Use the first NdArray
            match ys.swap_remove(selector) {
                OpOutput::Owned(y) => Ok(y),
                OpOutput::View(y) => Ok(unsafe { y.deref_into_view() }.to_owned())
            }
        })
    }
}

fn find_placeholder_value_by_key<'feeds, F: Float>(
    feeds: &'feeds [Feed<F>],
    node: &Tensor<F>,
    placeholder_name: &str,
) -> NdArrayView<'feeds, F> {
    for feed in feeds {
        match feed.placeholder_key {
            PlaceholderKey::ID(id) => {
                if node.id == id {
                    let ret = feed.value.view();
                    node.validate_using_known_shape(ret.shape());
                    return ret;
                }
            }
            PlaceholderKey::Name(name) => {
                if placeholder_name == name {
                    let ret = feed.value.view();
                    node.validate_using_known_shape(ret.shape());
                    return ret;
                }
            }
        }
    }
    panic!("Placeholder unfilled");
}

impl<F: Float> Graph<F> {
    pub(crate) fn eval<'feed, 'graph, 'tensor, A>(
        &'graph self,
        tensors: &'tensor [A],
        feeds: &[Feed<'feed, F>],
        env: &VariableEnvironment<F>,
    ) -> Vec<Result<NdArray<F>, crate::EvalError>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let mut storage = OpOutputStorage::new();

        // Graph traversal using depth-first-search
        // Vec<(tensor_id, should_visit)>
        let mut dfs_stack = Vec::<(TensorID, bool)>::with_capacity(1 << 10);

        for t in tensors {
            crate::graph::assert_same_graph(self, t.as_ref().graph);
            dfs_stack.push((t.as_ref().id(), false));
        }

        while let Some((node_id, should_visit)) = dfs_stack.pop() {
            let target_node = self.access_inner(node_id);

            if should_visit {
                if would_not_visit(&target_node, &storage) {
                    continue;
                }

                // ===========================================
                // Aggregate input values for the `target_node`
                // ===========================================

                // input arrays for `Op::compute`
                let mut op_inputs = op::SmallVec::new();

                // Would be Err if fail to collect input arrays
                let mut incoming_nodes_status = Ok(());

                // Initialize `op_inputs`
                for incoming in &target_node.incoming_nodes {
                    let in_tensor = incoming.as_tensor(self);
                    let in_ndarray = {
                        if let Some(ph_name) = in_tensor.placeholder_name() {
                            // use placeholder
                            Ok(OpInput::new_non_variable(find_placeholder_value_by_key(feeds, &in_tensor, ph_name)))
                        } else if let Some(vid) = incoming.get_variable_id(self) {
                            // use variable
                            if incoming.allow_mut {
                                Ok(OpInput::new_rdwr_variable(env.as_view_mut(vid)))
                            } else {
                                Ok(OpInput::new_rdonly_variable(env.as_view(vid)))
                            }
                        } else {
                            storage.get(incoming.id, incoming.array_selector).and_then(|got| {
                                Ok(OpInput::new_non_variable(unsafe { got.deref_into_view()}))
                            })
                        }
                    };
                    match in_ndarray {
                        Ok(x) => op_inputs.push(x),
                        Err(e) => {
                            incoming_nodes_status = Err(e);
                            break;
                        }
                    }
                }

                // =================
                // Run Op::compute()
                // =================
                let compute_result = incoming_nodes_status.and_then(|()| {
                    let mut op_ctx = op::ComputeContext::new(op_inputs);
                    let compute_status = target_node.get_op().compute(&mut op_ctx);
                    debug_assert!(!op_ctx.ys.is_empty(), "Bad op implementation: empty return value");
                    let output = compute_status.map(|()| op_ctx.ys);
                    output
                });
                storage.insert(node_id, compute_result);
            } else {
                // Update dfs stack
                dfs_stack.push((node_id, true));
                // Push children if needed
                for child in &target_node.incoming_nodes {
                    let child = self.access_inner(child.id);
                    if !would_not_visit(&child, &storage) {
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
                Ok(env.array_list[vid.0].clone().into_inner())
            } else if let Some(name) = t.placeholder_name() {
                // case 2: placeholder tensor
                Ok(find_placeholder_value_by_key(feeds, t, name).to_owned())
            } else {
                // case 3: normal tensor
                storage.take(t.id, 0).map_err(|e| {
                    crate::EvalError::OpError(e)
                })
            };
            ret.push(arr);
        }
        ret
    }
}

#[inline]
fn would_not_visit<F: Float>(
    node: &Ref<TensorInternal<F>>,
    storage: &OpOutputStorage<F>,
) -> bool {
    node.placeholder_name.is_some() || node.is_variable() || storage.inner.contains_key(&node.id())
}

#[test]
fn test_eval2() {
    use crate::tensor_ops as T;

    let ctx = VariableEnvironment::new();
    ctx.run(|g: &mut Context<f32>| {
        let a = T::ones(&[1, 1], g);
        let b = T::sigmoid(a);
        b.eval(g).unwrap();
    })
}

#[test]
fn test_eval() {
    use crate::tensor_ops as T;

    let ctx = VariableEnvironment::new();
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
    let ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let arr = ndarray::arr1(&[0., 0., 0.]).into_dyn();
        assert_eq!(Ok(arr.clone()), convert_to_tensor(arr, g).eval(g));
    });
}

#[test]
fn test_placeholder_eval() {
    // Needed for .strides() method

    let ctx = VariableEnvironment::new();
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
    let ctx = VariableEnvironment::new();
    ctx.run(|g| {
        let v: Tensor<f32> = g.placeholder("v", &[3, 2, 1]);
        let v2: Tensor<f32> = g.placeholder("v2", &[3, 2, 1]);
        let b = v + v2;
        let _results = g
            .evaluator()
            .push(b)
            .feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())
            .feed("v2", crate::ndarray_ext::ones(&[3, 2, 1]).view())
            .run();
    })
}
