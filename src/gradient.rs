//! Defining things related to gradient computation.
use crate::op::{GradientContext, InputArray};
use crate::tensor::{Tensor, TensorInternal};
use crate::Float;
use crate::FxHashMap;
use crate::Graph;
use std::cell::UnsafeCell;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

struct GradInfo<'a, 'b, T: Float + 'a> {
    has_gradient: bool,
    grad_called: bool,
    computed_grads: UnsafeCell<InputArray<Tensor<'b, T>>>,
    accumulated_grad: UnsafeCell<Option<Tensor<'b, T>>>,
    default_grad: Option<&'a TensorInternal<T>>,
}

impl<'t, 's, T: Float> GradInfo<'t, 's, T> {
    #[inline]
    fn new(has_gradient: bool, default_grad: Option<&'t TensorInternal<T>>) -> GradInfo<'t, 's, T> {
        GradInfo {
            has_gradient,
            computed_grads: UnsafeCell::new(InputArray::new()),
            grad_called: false,
            accumulated_grad: UnsafeCell::new(None),
            default_grad,
        }
    }

    #[inline]
    fn push_grad(&self, g: Tensor<'s, T>) {
        unsafe {
            (&mut *self.computed_grads.get()).push(g);
        }
    }

    fn accumulate_then_get(&self, s: &'s Graph<T>) -> Tensor<'s, T> {
        unsafe {
            if let Some(ret) = *self.accumulated_grad.get() {
                // accumulation completed
                ret
            } else {
                let before_acc = &*self.computed_grads.get();
                if before_acc.len() > 1 {
                    // accumulate
                    let accumulated = s.add_n(before_acc.as_slice());
                    *self.accumulated_grad.get() = Some(accumulated);
                    accumulated
                } else {
                    // accumulation is not required
                    before_acc[0]
                }
            }
        }
    }
}

#[inline]
fn has_marked_child<'a, 'b, T: Float>(
    s: &'b Graph<T>,
    parent: &TensorInternal<T>,
    path: &FxHashMap<usize, GradInfo<'a, 'b, T>>,
) -> bool {
    for child in parent.get_backprop_inputs().iter() {
        if path.get(&child.get(s).id()).unwrap().has_gradient {
            return true;
        }
    }
    false
}

#[inline]
fn is_wrt<T: Float>(node: &TensorInternal<T>, wrt: &[&TensorInternal<T>]) -> bool {
    wrt.contains(&node)
}

// Marks `has_gradient` if each node is on the gradient propagation path.
// Map keys of return value are tensor ids.
//
// Strategy
//   1. Record all nodes that are reachable from `ys` into `ret`.
//   2. Mark the path between `ys` and `xs` as `has_gradient`.
fn get_between_nodes<'a, 'b: 'a, T: Float>(
    s: &'b Graph<T>,
    ys: &[&'a TensorInternal<T>],
    wrt: &[&'a TensorInternal<T>],
) -> FxHashMap<usize, GradInfo<'a, 'b, T>> {
    // Randomly accessible by use of each node's lookup key.
    let mut ret = FxHashMap::<usize, GradInfo<T>>::default();

    // Builds GradInfo while performing depth-first-search.
    // `has_gradient` properties are filled at the same time.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(&TensorInternal<T>, bool)> = ys.iter().map(|&y| (y, false)).collect();
    while let Some((node, should_visit)) = dfs_stack.pop() {
        if should_visit {
            let marker =
                node.is_differentiable && (is_wrt(node, wrt) || has_marked_child(s, node, &ret));
            ret.insert(node.id(), GradInfo::new(marker, None));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((node, true));
            // Push children as necessary
            for child in node.get_backprop_inputs() {
                let child = child.get(s);
                if ret.get(&node.id()).is_none() {
                    if child.is_source() || !child.is_differentiable() {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `wrt` nodes in this direction....
                        ret.insert(
                            child.id(),
                            GradInfo::new(
                                child.is_differentiable() && is_wrt(child.inner(), wrt),
                                None,
                            ),
                        );
                    } else {
                        // Recurse
                        dfs_stack.push((child.scoped_inner(), false));
                    }
                }
            }
        }
    }
    ret
}

/// Returns symbolic gradient tensors of `xs`.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building a subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `gys` are already known gradients of `ys`'s outputs.
///
/// NOTE: Nodes that do not have gradients won't be included in the subgraph to avoid
/// unnecessary computation.
pub(crate) fn symbolic_gradients<'a, 'b, T: Float>(
    ys: &[&'a TensorInternal<T>],
    wrt: &[&'a TensorInternal<T>],
    gys: &[&'a TensorInternal<T>],
    g: &'b Graph<T>,
) -> Vec<Tensor<'b, T>> {
    assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");

    // Setup gradient path.
    // We lookup this with tensor id.
    let mut between_nodes = get_between_nodes(g, ys, wrt);

    // Set default grads.
    for (y, gy) in ys.iter().zip(gys) {
        between_nodes.get_mut(&y.id()).unwrap().default_grad = Some(gy);
    }

    // Prepare a heap with given ys.
    let mut heap = ys
        .iter()
        .map(|y| y.wrapped())
        .collect::<BinaryHeap<TensorWrapper<T>>>();

    // Backprop.
    // Starts with `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let info: &GradInfo<_> = between_nodes.get(&y.tsr.id()).unwrap();
            let gy = if let Some(def) = info.default_grad {
                g.scoped(def)
            } else {
                info.accumulate_then_get(g)
            };
            // Call Op::grad
            let mut ctx = GradientContext::new(gy, g.scoped(y.tsr), g);
            y.tsr.op.grad(&mut ctx);
            let gxs = ctx.extract_input_grads();
            debug_assert_eq!(y.tsr.in_edges.len(), gxs.len());
            gxs
        };
        // Register computed gradients
        let xs = y.tsr.get_backprop_inputs();
        for (gx, x) in gxs.into_iter().zip(xs) {
            let x = x.get(g);
            let mut x_info = between_nodes.get_mut(&x.id()).unwrap();
            if x_info.has_gradient {
                if let Some(gx) = gx {
                    x_info.push_grad(gx);
                    // update heap
                    if !x.is_source() && !x_info.grad_called {
                        x_info.grad_called = true;
                        heap.push(x.scoped_inner().wrapped());
                    }
                }
            }
        }
    }

    // Aggregate and return xs's gradients
    let mut ret = Vec::with_capacity(wrt.len());
    for x in wrt {
        let msg1: &str = "Not differentiable with given tensor(s).";
        let info = between_nodes.get(&x.id()).expect(msg1);
        if !info.has_gradient {
            panic!(msg1);
        }
        assert!(
            info.default_grad.is_none(),
            "Can't differentiate with objective itself"
        );
        ret.push(info.accumulate_then_get(g));
    }
    ret
}

struct TensorWrapper<'a, T: Float + 'a> {
    tsr: &'a TensorInternal<T>,
}

impl<'a, T: Float> Ord for TensorWrapper<'a, T> {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.tsr.top_rank.cmp(&other.tsr.top_rank)
    }
}

impl<'a, T: Float> PartialOrd for TensorWrapper<'a, T> {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.tsr.top_rank.cmp(&other.tsr.top_rank))
    }
}

impl<'a, T: Float> Eq for TensorWrapper<'a, T> {}

impl<'a, T: Float> PartialEq for TensorWrapper<'a, T> {
    #[inline]
    fn eq(&self, other: &TensorWrapper<'a, T>) -> bool {
        self.tsr.id() == other.tsr.id()
    }
}

impl<'a, T: Float> TensorInternal<T> {
    #[inline]
    fn wrapped(&'a self) -> TensorWrapper<'a, T> {
        TensorWrapper { tsr: self }
    }
}
