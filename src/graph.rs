//! Defining things related to `ag::Graph`.

use crate::tensor::{Tensor, TensorInternal};

use crate::variable::{VariableID, VariableNamespace};
use crate::{Float, FxHashMap, NdArray, VariableEnvironment};

use std::cell::RefCell;
use std::cell::{Ref, RefMut};
use std::fmt;
use std::ops::Deref;

type TensorID = usize;

pub const NUM_NODES_WARN: usize = 50_000;
pub const NUM_NODES_CRITICAL: usize = 500_000;

/// Holds tensors inside.
///
/// NOTE:
/// You won't be using `Graph` struct directly because this is generally accessed via `Context::deref`.
pub struct Graph<F: Float> {
    pub(crate) node_set: RefCell<Vec<TensorInternal<F>>>,
    pub(crate) variable2node: RefCell<FxHashMap<VariableID, TensorID>>,
}

impl<'t, 'g, F: Float> Graph<F> {
    #[inline]
    pub(crate) fn install(&'g self, mut node: TensorInternal<F>) -> TensorID {
        let mut inner = self.node_set.borrow_mut();
        let id = inner.len();
        if id == NUM_NODES_WARN {
            eprintln!(
                "Too many tensors in this graph: {}. \
            Use Graph::clear, or stop using loops in the VariableEnvironment::run block",
                NUM_NODES_WARN
            )
        }
        if id > NUM_NODES_CRITICAL {
            panic!(
                "Maximum graph size exceeded: {}. \
            Use Graph::clear, or stop using loops in the VariableEnvironment::run block",
                NUM_NODES_CRITICAL
            )
        }
        node.id = id;
        inner.push(node);
        id
    }

    #[inline(always)]
    pub(crate) fn access_inner(&self, i: TensorID) -> Ref<TensorInternal<F>> {
        let borrow = self.node_set.borrow();
        Ref::map(borrow, |t| &t[i])
    }

    #[inline(always)]
    pub(crate) fn access_inner_mut(&self, i: TensorID) -> RefMut<TensorInternal<F>> {
        let borrow = self.node_set.borrow_mut();
        RefMut::map(borrow, |t| &mut t[i])
    }

    #[inline(always)]
    pub(crate) fn tensor(&'g self, id: TensorID) -> Tensor<'g, F> {
        Tensor { id, graph: self }
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
    let env_handle = &mut VariableEnvironment::new();
    let graph_internal = Graph {
        node_set: RefCell::new(Vec::with_capacity(512)),
        variable2node: RefCell::new(FxHashMap::default()),
    };
    let mut g = Context {
        env_handle,
        inner: graph_internal,
    };
    f(&mut g)
}

/// Context for creating and evaluating tensors
///
/// Use [run] or [VariableEnvironment::run] to instantiate this.
///
/// ```
/// use autograd as ag;
/// use ag::ndarray;
/// use ag::tensor_ops as T;
///
/// let grad = ag::run(|ctx| {
///     let x = ctx.placeholder("x", &[]);
///     let y = ctx.placeholder("y", &[]);
///     let z = 2.*x*x + 3.*y + 1.;
///
///     // dz/dx (symbolic):
///     let grad = &T::grad(&[z], &[x])[0];
///
///     // Evaluate dz/dx when x=3:
///     ctx.evaluator()
///         .push(grad)
///         .feed(x, ndarray::arr0(3.0).view())
///         .run().remove(0)
/// });
/// assert_eq!(grad.unwrap(), ndarray::arr0(12.0).into_dyn());
/// ```
pub struct Context<'env, 'name, F: Float> {
    pub(crate) env_handle: &'env VariableEnvironment<'name, F>,
    pub(crate) inner: Graph<F>,
}

impl<'g, 'env, 'name, F: Float> Context<'env, 'name, F> {
    /// Get or create a namespace with the specified name.
    ///
    /// Use `namespace_mut` for mutable usages such as variables registrations.
    #[inline]
    pub fn namespace(&'env self, namespace_id: &'static str) -> VariableNamespace<'env, 'name, F> {
        self.env_handle.namespace(namespace_id)
    }

    /// Get or create the *default* namespace.
    ///
    /// Use `namespace_mut` for mutable usages such as variables registrations.
    #[inline]
    pub fn default_namespace(&'env self) -> VariableNamespace<'env, 'name, F> {
        self.env_handle.default_namespace()
    }

    /// Returns a reference to the current VariableEnvironment
    #[inline]
    pub fn env(&'g self) -> &'env VariableEnvironment<F> {
        self.env_handle
    }

    /// Removes all tensors in this graph.
    ///
    /// Note that any tensors allocated prior to this method call are invalid.
    #[inline]
    pub fn clear(&mut self) {
        self.inner.node_set.borrow_mut().clear();
        self.inner.variable2node.borrow_mut().clear();
    }

    /// Creates a placeholder tensor.
    ///
    /// Behaves like TensorFlow 1.x 's placeholder.
    /// `shape_[i]` must be a positive value, or -1 which means dynamic dim.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::run(|c| {
    ///     // x1 and x2 represent the same value
    ///     let x1 = c.placeholder("x", &[-1, 2]);
    ///     let x2 = c.placeholder("x", &[-1, 2]);
    ///     let sum = x1 + x2;
    ///
    ///     let arr = &ag::ndarray::array![[1., 1.]].into_dyn();
    ///     let ret = c.evaluator()
    ///         .push(&sum)
    ///         .feed("x", arr.view()) // feed for x1 and x2
    ///         .run();
    ///     assert_eq!(ret[0], Ok(arr + arr));
    ///
    ///     // Same result
    ///     let ret = c.evaluator()
    ///         .push(sum)
    ///         .feed(x1, arr.view())
    ///         .feed(x2, arr.view())
    ///         .run();
    ///     assert_eq!(ret[0], Ok(arr + arr));
    /// });
    /// ```
    ///
    /// See also [crate::evaluation::Evaluator] example.
    #[inline]
    pub fn placeholder(&'g self, name: &'static str, shape_: &[isize]) -> Tensor<'g, F> {
        use crate::tensor_ops as T;
        let b = Tensor::builder(self).set_placeholder_name(name);
        let rank = shape_.len();
        let b = if rank == 0 || -1 != shape_[0] {
            let shape = T::convert_to_tensor(
                NdArray::from_shape_vec(
                    ndarray::IxDyn(&[rank]),
                    shape_
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
        let b = b.set_known_shape(shape_);
        b.build(T::basic_source_ops::Placeholder)
    }
}

impl<'env, 'name, F: Float> Deref for Context<'env, 'name, F> {
    type Target = Graph<F>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.inner
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

impl<F: Float> AsGraph<F> for Context<'_, '_, F> {
    #[inline]
    fn as_graph(&self) -> &Graph<F> {
        &self.inner
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
    crate::VariableEnvironment::<f32>::new().run(|g| {
        let a = T::zeros(&[1], g);
        crate::VariableEnvironment::<f32>::new().run(|g2| {
            let b = T::zeros(&[1], g2);
            let _ = a + b;
        });
    });
}
