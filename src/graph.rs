use crate::tensor::{Tensor, TensorInternal};

use crate::{Evaluator, Feeder, tensor_ops as T, tensor_ops};
use crate::variable::{VariableID, VariableNamespace};
use crate::{Float, FxHashMap, NdArray, VariableEnvironment};

use std::cell::{Ref, RefMut, RefCell};
use std::fmt;
use std::ops::Deref;

pub type TensorID = usize;

/// Graph represents a computation graph holding tensors inside.
///
/// NOTE:
/// You won't be using this struct directly because this is generally accessed via `Context::deref()`.
pub struct Graph<F: Float> {
    pub(crate) node_set: RefCell<Vec<TensorInternal<F>>>,
    pub(crate) variable2node: RefCell<FxHashMap<VariableID, TensorID>>,
}

pub const NUM_NODES_WARN: usize = 50_000;
pub const NUM_NODES_CRITICAL: usize = 500_000;

impl<'graph, F: Float> Graph<F> {
    #[inline]
    pub(crate) fn install(&'graph self, mut node: TensorInternal<F>) -> TensorID {
        let mut inner = self.node_set.borrow_mut();
        let id = inner.len();
        if id == NUM_NODES_WARN {
            eprintln!(
                "Too many tensors in this graph: {}. \
            Use Graph::clear, or move the training loop out of the `run` block",
                NUM_NODES_WARN
            )
        }
        if id > NUM_NODES_CRITICAL {
            panic!(
                "Maximum graph size exceeded: {}. \
            Use Graph::clear, or move the training loop out of the `run` block",
                NUM_NODES_CRITICAL
            )
        }
        node.id = id;
        inner.push(node);
        id
    }

    #[inline(always)]
    pub(crate) fn access_inner(&self, id: TensorID) -> Ref<TensorInternal<F>> {
        let borrow = self.node_set.borrow();
        Ref::map(borrow, |t| &t[id])
    }

    #[inline(always)]
    pub(crate) fn access_inner_mut(&self, id: TensorID) -> RefMut<TensorInternal<F>> {
        let borrow = self.node_set.borrow_mut();
        RefMut::map(borrow, |t| &mut t[id])
    }

    #[inline(always)]
    pub(crate) fn tensor(&'graph self, id: TensorID) -> Tensor<'graph, F> {
        Tensor { id, graph: self }
    }

    #[inline]
    pub(crate) fn topo_rank(&self, id: TensorID) -> usize {
        self.node_set.borrow()[id].topo_rank
    }
}

impl<T: Float> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let set = &*self.node_set.borrow();
        let mut buf = format!("graph size: {}\n", set.len());
        for node in set {
            buf += format!("{}\n", node).as_str();
        }
        write!(f, "{}", buf)
    }
}

/// Creates and runs a computation graph.
///
/// See [Context].
pub fn run<F, FN, R>(f: FN) -> R
where
    F: Float,
    FN: FnOnce(&mut Context<F>) -> R,
{
    let graph_internal = Graph {
        node_set: RefCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(FxHashMap::default()),
    };
    let mut ctx = Context {
        var_env_ref: &mut VariableEnvironment::new(),
        graph: graph_internal,
    };
    f(&mut ctx)
}

/// Generates and runs a computation graph
///
/// Each time [run] is invoked, a new `Context` allocating a [Graph] is passed to the closure, in which tensors are generated and evaluated.
/// It's faster to understand if you see [Tensor]'s documentation.
///
/// In order to bind `Tensor`s to pre-defined variable arrays, use [VariableEnvironment::run] instead.
/// See [crate::variable]
pub struct Context<'env, F: Float> {
    pub(crate) graph: Graph<F>,
    pub(crate) var_env_ref: &'env VariableEnvironment<F>,
}

impl<'graph, 'env, F: Float> Context<'env, F> {
    /// Get or create a variable namespace with the specified name.
    ///
    /// Use `namespace_mut` for mutable operations such as variables registrations.
    #[inline]
    pub fn namespace(&'env self, namespace_id: &'static str) -> VariableNamespace<'env, F> {
        self.var_env_ref.namespace(namespace_id)
    }

    /// Get or create the *default* variable namespace.
    ///
    /// Use `namespace_mut` for mutable operations such as variables registrations.
    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, F> {
        self.var_env_ref.default_namespace()
    }

    /// Returns a reference to the current VariableEnvironment
    #[inline]
    pub fn env(&'graph self) -> &'env VariableEnvironment<F> {
        self.var_env_ref
    }

    /// Removes all tensors in this graph.
    ///
    /// Note that any tensors allocated prior to this method call are invalid.
    #[inline]
    pub fn clear(&mut self) {
        self.graph.node_set.borrow_mut().clear();
        self.graph.variable2node.borrow_mut().clear();
    }

    /// Creates a placeholder tensor in a [Graph].
    ///
    /// placeholder is a named tensor whose value can be specified when evaluating a computation graph.
    /// You can designate the `shape` of the placeholder and `shape[i]` can be a positive
    /// value or -1 which means an dim of arbitrary size.
    ///
    /// Use [Evaluator::feed] and [Feeder::push] in order to assign ArrayViews to placeholders.
    /// ```
    /// use autograd as ag;
    /// use ag::ndarray::array;
    ///
    /// ag::run(|ctx| {
    ///     // be aware that x1 and x3 represent the same value
    ///     let x1 = ctx.placeholder("x", &[-1, 2]);
    ///     let x2 = ctx.placeholder("y", &[-1, 2]);
    ///     let x3 = ctx.placeholder("x", &[-1, 2]);
    ///     let sum = x1 + x2 + x3;
    ///
    ///     let arr = &array![[1., 1.]].into_dyn();
    ///
    ///     let result = ctx.evaluator()
    ///         .push(&sum)
    ///         .feed("x", arr.view()) // feed for x1 and x3
    ///         .feed("y", arr.view()) // feed for x2
    ///         .feed(x2, arr.view()) // same as .feed("y", ...)
    ///         .run();
    ///     assert_eq!(result[0], Ok(arr + arr + arr));
    /// });
    /// ```
    ///
    /// See also [tensor_ops::convert_to_tensor].
    #[inline]
    pub fn placeholder(&'graph self, name: &'static str, shape: &[isize]) -> Tensor<'graph, F> {
        let b = Tensor::builder(self).set_placeholder_name(name);
        let rank = shape.len();
        let b = if rank == 0 || -1 != shape[0] {
            let shape = T::convert_to_tensor(
                NdArray::from_shape_vec(
                    ndarray::IxDyn(&[rank]),
                    shape
                        .iter()
                        .map(|&x| F::from(x).unwrap())
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
                self,
            );
            b.set_shape(&shape)
        } else {
            b
        };
        let b = b.set_known_shape(shape);
        b.build(T::basic_source_ops::Placeholder)
    }
}

impl<'env, F: Float> Deref for Context<'env, F> {
    type Target = Graph<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.graph
    }
}

pub trait AsGraph<F: Float> {
    fn as_graph(&self) -> &Graph<F>;
}

impl<F: Float> AsGraph<F> for Graph<F> {
    #[inline]
    fn as_graph(&self) -> &Graph<F> {
        self
    }
}

impl<F: Float> AsGraph<F> for Context<'_, F> {
    #[inline]
    fn as_graph(&self) -> &Graph<F> {
        &self.graph
    }
}

#[inline]
pub(crate) fn assert_same_graph<F: Float>(a: &impl AsGraph<F>, b: &impl AsGraph<F>) {
    assert_eq!(
        a.as_graph() as *const _,
        b.as_graph() as *const _,
        "Detected tensors belonging to different graphs"
    );
}

#[test]
#[should_panic]
fn test_mixed_graph() {
    VariableEnvironment::<f32>::new().run(|g| {
        let a = T::zeros(&[1], g);
        VariableEnvironment::<f32>::new().run(|g2| {
            let b = T::zeros(&[1], g2);
            let _ = a + b;
        });
    });
}
