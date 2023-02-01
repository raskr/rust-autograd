//! Defining things related to gradient computation.
use crate::graph::TensorID;
use crate::op::{GradientContext, SmallVec};
use crate::tensor::Tensor;
use crate::tensor_ops as T;
use crate::Float;
use crate::FxHashMap;
use crate::Graph;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;

/// Returns gradient tensors of `xs`.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building a subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `gys` are already known gradients of `ys`'s outputs.
///
/// NOTE:
/// Returned gradient is `None` if the corresponding variable is not differentiable.
pub(crate) fn compute_gradients<'graph, A, B, F: Float>(
    ys: &[A],
    xs: &[B],
    gys: Option<&[Tensor<'graph, F>]>,
    g: &'graph Graph<F>,
) -> GradientMap<'graph, F>
where
    A: AsRef<Tensor<'graph, F>>,
    B: AsRef<Tensor<'graph, F>>,
{
    let mut grad_map = make_gradient_map(g, ys, xs);

    // Set default grads.
    if let Some(gys) = gys {
        assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");
        for (y, gy) in ys.into_iter().zip(gys) {
            grad_map.get_mut(&y.as_ref().id).unwrap().default_grad = Some(gy.as_ref().clone());
        }
    } else {
        let start_gy = T::scalar(F::one(), g);
        for y in ys.into_iter() {
            grad_map.get_mut(&y.as_ref().id).unwrap().default_grad = Some(start_gy.clone());
        }
    }

    // Prepare a heap with given ys.
    let mut heap = ys
        .iter()
        .map(|y| y.as_ref().to_node())
        .collect::<BinaryHeap<Node>>();

    // Backprop.
    // Starts with `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let g_info = grad_map.get_mut(&y.id).unwrap();

            let gy = g_info.get_grad_tensor();

            // Call Op::grad (mutate the graph)
            let y_tensor = g.tensor(y.id);
            let gxs = GradientContext::new(gy, y_tensor, g).compute_input_grads();
            debug_assert_eq!(y_tensor.num_backprop_inputs(), gxs.len());
            gxs
        };

        // Register computed gradients
        let y = g.tensor(y.id);
        for (i, x) in y.inner().get_backprop_inputs().iter().enumerate() {
            let x = x.as_tensor(g);
            let mut x_info = grad_map.get_mut(&x.id).unwrap();
            if x_info.on_gradient_path {
                if let Some(gx) = gxs[i] {
                    x_info.push_grad(gx);
                    // update heap
                    if !x.is_source() && !x_info.grad_computed {
                        x_info.grad_computed = true;
                        heap.push(x.to_node());
                    }
                }
            }
        }
    }

    GradientMap {
        inner: grad_map,
    }
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

impl<'tensor, T: Float> Tensor<'tensor, T> {
    #[inline]
    fn to_node(&'tensor self) -> Node {
        Node {
            id: self.id,
            rank: self.inner().topo_rank,
        }
    }
}

// compute_gradients's return value
pub(crate) struct GradientMap<'graph, F: Float> {
    inner: FxHashMap<TensorID, GradientMapValue<'graph, F>>,
}

impl<'graph, F: Float> GradientMap<'graph, F> {
    #[inline]
    pub(crate) fn get(&mut self, x: impl AsRef<Tensor<'graph, F>>) -> Option<Tensor<'graph, F>> {
        if let Some(info) = self.inner.get_mut(&x.as_ref().id) {
            if info.on_gradient_path && info.default_grad.is_none() {
                return Some(info.accumulate_then_get());
            }
        }
        // can't differentiate!
        None
    }
}

// Return value of compute_gradients()
// Used like HashMap<TensorID, GradInfo>
struct GradientMapValue<'graph, F: Float> {
    on_gradient_path: bool,
    grad_computed: bool,
    computed_grads: SmallVec<Tensor<'graph, F>>,
    accumulated_grad: Option<Tensor<'graph, F>>,
    default_grad: Option<Tensor<'graph, F>>,
}

impl<'graph, F: Float> GradientMapValue<'graph, F> {
    #[inline]
    fn new(has_gradient: bool) -> GradientMapValue<'graph, F> {
        GradientMapValue {
            on_gradient_path: has_gradient,
            computed_grads: SmallVec::new(),
            grad_computed: false,
            accumulated_grad: None,
            default_grad: None,
        }
    }

    #[inline]
    fn push_grad(&mut self, g: Tensor<'graph, F>) {
        self.computed_grads.push(g);
    }

    fn accumulate_then_get(&mut self) -> Tensor<'graph, F> {
        if let Some(acc) = self.accumulated_grad {
            return acc;
        }
        if self.computed_grads.len() == 1 {
            self.computed_grads[0]
        } else {
            // accumulation is required
            let accumulated = T::add_n(self.computed_grads.as_slice());
            self.accumulated_grad = Some(accumulated);
            accumulated
        }
    }

    #[inline]
    fn get_grad_tensor(&mut self) -> Tensor<'graph, F> {
        if let Some(def) = self.default_grad {
            def
        } else {
            self.accumulate_then_get()
        }
    }
}

#[inline]
fn has_child_on_path<T: Float>(parent: Tensor<T>, path: &FxHashMap<usize, GradientMapValue<T>>) -> bool {
    let inner = parent.inner();
    for child in inner.get_backprop_inputs() {
        if path.get(&child.id).unwrap().on_gradient_path {
            return true;
        }
    }
    false
}

// checks `candidate` node is an xs node or not.
#[inline]
fn is_given_xs<'graph, F: Float, A>(candidate: usize, xs: &[A]) -> bool
    where
        A: AsRef<Tensor<'graph, F>>,
{
    for x in xs {
        if x.as_ref().id == candidate {
            return true;
        }
    }
    false
}

// Go backward from ys and collect reachable nodes.
// Nodes between `ys` and `xs` are marked as `on_gradient_path`.
fn make_gradient_map<'graph, A, B, F: Float>(
    g: &'graph Graph<F>,
    ys: &[A],
    xs: &[B],
) -> FxHashMap<TensorID, GradientMapValue<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>>,
        B: AsRef<Tensor<'graph, F>>,
{
    let mut ret = FxHashMap::<TensorID, GradientMapValue<F>>::default();

    // Builds GradInfo while performing depth-first-search.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(TensorID, bool)> = ys.iter().map(|y| (y.as_ref().id, false)).collect();
    while let Some((curr_id, should_visit)) = dfs_stack.pop() {
        let curr_node = g.tensor(curr_id);
        if should_visit {
            let on_grad_path =
                curr_node.is_differentiable() && (is_given_xs(curr_id, xs) || has_child_on_path(curr_node, &ret));
            ret.insert(curr_id, GradientMapValue::new(on_grad_path));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((curr_id, true));
            // Push children as necessary
            let curr_node = curr_node.inner();
            for child in curr_node.get_backprop_inputs() {
                let child = child.as_tensor(g);
                if ret.get(&curr_id).is_none() {
                    if child.is_source() || !child.is_differentiable() {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `xs` nodes in this direction....
                        ret.insert(
                            child.id,
                            GradientMapValue::new(child.is_differentiable() && is_given_xs(child.id, xs)),
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

