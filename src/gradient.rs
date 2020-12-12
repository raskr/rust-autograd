//! Defining things related to gradient computation.
use crate::op::{GradientContext, InputArray};
use crate::tensor::Tensor;
use crate::Float;
use crate::FxHashMap;
use crate::Graph;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

// Info of gradient of a `Tensor`.
struct GradInfo<'graph, T: Float> {
    has_gradient: bool,
    grad_called: bool,
    computed_grads: InputArray<Tensor<'graph, T>>,
    accumulated_grad: Option<Tensor<'graph, T>>,
    default_grad: Option<usize>, // id
}

impl<'g, T: Float> GradInfo<'g, T> {
    #[inline]
    fn new(has_gradient: bool) -> GradInfo<'g, T> {
        GradInfo {
            has_gradient,
            computed_grads: InputArray::new(),
            grad_called: false,
            accumulated_grad: None,
            default_grad: None,
        }
    }

    #[inline]
    fn push_grad(&mut self, g: Tensor<'g, T>) {
        self.computed_grads.push(g);
    }

    #[inline]
    fn accumulate_then_get(&mut self, g: &'g Graph<T>) -> Tensor<'g, T> {
        if let Some(acc) = self.accumulated_grad {
            return acc;
        }
        if self.computed_grads.len() == 1 {
            self.computed_grads[0]
        } else {
            // accumulation is required
            let accumulated = g.add_n(self.computed_grads.as_slice());
            self.accumulated_grad = Some(accumulated);
            accumulated
        }
    }

    #[inline]
    fn get_grad(&mut self, g: &'g Graph<T>) -> Tensor<'g, T> {
        if let Some(def) = self.default_grad {
            g.tensor(def)
        } else {
            self.accumulate_then_get(g)
        }
    }
}

#[inline]
fn has_marked_child<T: Float>(parent: Tensor<T>, path: &FxHashMap<usize, GradInfo<T>>) -> bool {
    for i in 0..parent.num_backprop_inputs() {
        let child = parent.get_backprop_input(i);
        if path.get(&child.id).unwrap().has_gradient {
            return true;
        }
    }
    false
}

#[inline]
fn is_wrt(node: usize, wrt: &[usize]) -> bool {
    wrt.contains(&node)
}

// Go backward from `ys` and collect nodes until reach `wrt` for backprop.
//
// Strategy
//   1. Record all nodes that are reachable from `ys` into `ret`.
//   2. Mark the path between `ys` and `xs` as `has_gradient`.
fn get_between_nodes<'t, 'g, T: Float>(
    g: &'g Graph<T>,
    ys: &[usize],
    wrt: &[usize],
) -> FxHashMap<usize, GradInfo<'g, T>> {
    // Randomly accessible by use of each node's id.
    let mut ret = FxHashMap::<usize, GradInfo<T>>::default();

    // Builds GradInfo while performing depth-first-search.
    // `has_gradient` properties are filled at the same time.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(usize, bool)> = ys.iter().map(|&y| (y, false)).collect();
    while let Some((node_id, should_visit)) = dfs_stack.pop() {
        let node = g.tensor(node_id);
        if should_visit {
            let has_gradient =
                node.is_differentiable() && (is_wrt(node_id, wrt) || has_marked_child(node, &ret));
            ret.insert(node_id, GradInfo::new(has_gradient));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((node_id, true));
            // Push children as necessary
            for i in 0..node.num_backprop_inputs() {
                let child = node.get_backprop_input(i).as_tensor(g);
                if ret.get(&node_id).is_none() {
                    if child.is_source() || !child.is_differentiable() {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `wrt` nodes in this direction....
                        ret.insert(
                            child.id,
                            GradInfo::new(child.is_differentiable() && is_wrt(child.id, wrt)),
                        );
                    } else {
                        // Recurse
                        dfs_stack.push((child.id, false));
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
pub(crate) fn symbolic_gradients<'t, 'g, T: Float>(
    ys: &[usize],
    wrt: &[usize],
    gys: &[usize],
    g: &'g Graph<T>,
) -> Vec<Tensor<'g, T>> {
    assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");

    // Setup gradient path.
    // We lookup this with tensor id.
    let mut between_nodes = get_between_nodes(g, ys, wrt);

    // Set default grads.
    for (y, gy) in ys.iter().zip(gys) {
        between_nodes.get_mut(y).unwrap().default_grad = Some(*gy);
    }

    // Prepare a heap with given ys.
    let mut heap = ys
        .iter()
        .map(|&y| g.tensor(y).to_node())
        .collect::<BinaryHeap<Node>>();

    // Backprop.
    // Starts with `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let info = between_nodes.get_mut(&y.id).unwrap();

            let gy = info.get_grad(g);

            // Call Op::grad (mutate the graph)
            let y_tensor = g.tensor(y.id);
            let gxs = GradientContext::new(gy, y_tensor, g).extract_input_grads();
            debug_assert_eq!(y_tensor.num_backprop_inputs(), gxs.len());
            gxs
        };

        // Register computed gradients
        let y = g.tensor(y.id);
        for i in 0..gxs.len() {
            let x = y.get_backprop_input(i).as_tensor(g);
            let mut x_info = between_nodes.get_mut(&x.id).unwrap();
            if x_info.has_gradient {
                if let Some(gx) = gxs[i] {
                    x_info.push_grad(gx);
                    // update heap
                    if !x.is_source() && !x_info.grad_called {
                        x_info.grad_called = true;
                        heap.push(x.to_node());
                    }
                }
            }
        }
    }

    // Aggregate and return xs's gradients
    let mut ret = Vec::with_capacity(wrt.len());
    for x in wrt {
        let msg1: &str = "Not differentiable with given tensor(s).";
        let info = between_nodes.get_mut(x).expect(msg1);
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

struct Node {
    id: usize,
    rank: usize,
}

impl Ord for Node {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.rank.cmp(&other.rank)
    }
}

impl PartialOrd for Node {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.rank.cmp(&other.rank))
    }
}

impl Eq for Node {}

impl PartialEq for Node {
    #[inline]
    fn eq(&self, other: &Node) -> bool {
        self.id == other.id
    }
}

impl<'t, T: Float> Tensor<'t, T> {
    #[inline]
    fn to_node(&'t self) -> Node {
        Node {
            id: self.id,
            rank: unsafe { self.inner().top_rank },
        }
    }
}
