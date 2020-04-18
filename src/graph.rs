//! Defining things related to `ag::Graph`.

use crate::{tensor::Tensor, tensor::TensorInternal, Float};
use std::cell::UnsafeCell;
use std::fmt;

/// Generator of `Tensor` objects.
///
/// Use [autograd::with](fn.with.html) to instantiate this.
///
/// ```
/// use autograd as ag;
///
/// ag::with(|graph1: &mut ag::Graph<f32>| {
///     // Creating some nodes (tensors) in this graph.
///     let a = graph1.zeros(&[2, 3]);
///     let b = graph1.ones(&[2, 3]);
///
///     // Evaluate the tensors
///     (a + b).eval(&[]);
///
///     // Creating another scope (graph).
///     ag::with(|graph2: &mut ag::Graph<f32>| {
///         // `c` is valid only in graph2.
///         let c = graph2.zeros(&[3, 4]);
///
///         // Cross-scope access to what derived from `Graph` can't compile for now.
///
///         // graph1.zeros(&[2, 3])
///         // ^^^^^^ invalid access for `graph1`
///
///         // a + c
///         // ^ invalid access for `a` that belongs to ``graph1`
///     });
///     // tensors in graph2 destructed here.
/// });
/// // tensors in graph1 destructed here.
/// ```
pub struct Graph<F: Float> {
    node_set: UnsafeCell<Vec<TensorInternal<F>>>,
}

impl<'t, 'g, F: Float> Graph<F> {
    pub(crate) fn install(&'g self, mut node: TensorInternal<F>) -> usize {
        unsafe {
            let inner = &mut *self.node_set.get();
            let id = inner.len();
            node.id = id;
            inner.push(node);
            id
        }
    }

    // `i` must be an id returned by Graph::install
    #[allow(dead_code)]
    pub(crate) fn access(&self, i: usize) -> Tensor<F> {
        unsafe { (*self.node_set.get()).get_unchecked(i).tensor(self) }
    }

    // `i` must be an id returned by Graph::install
    #[inline]
    pub(crate) fn access_inner(&self, i: usize) -> &'t TensorInternal<F> {
        unsafe { (*self.node_set.get()).get_unchecked(i) }
    }
}

impl<T: Float> fmt::Debug for Graph<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            let set = &*self.node_set.get();
            let mut buf = format!("graph size: {}\n", set.len());
            for node in set {
                buf += format!("{}\n", node).as_str();
            }
            write!(f, "{}", buf)
        }
    }
}

/// Creates a scope for a computation graph.
///
/// This is the only way to create [Graph](struct.Graph.html) instances.
pub fn with<F, FN>(f: FN)
where
    F: Float,
    FN: FnOnce(&mut Graph<F>) -> () + Send,
{
    let mut g = Graph {
        node_set: UnsafeCell::new(Vec::with_capacity(128)),
    };
    f(&mut g);
}
