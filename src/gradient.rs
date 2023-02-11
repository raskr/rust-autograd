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
    let mut grad_map = init_gradient_map(g, ys, xs);

    // Setup default grads.
    if let Some(gys) = gys {
        assert_eq!(ys.len(), gys.len(), "`ys.len()` must match `gys.len()`");
        for (y, &gy) in ys.into_iter().zip(gys) {
            grad_map.push_grad(y.as_ref().id, gy);
        }
    } else {
        let start_gy = T::scalar(F::one(), g);
        for y in ys.into_iter() {
            grad_map.push_grad(y.as_ref().id, start_gy);
        }
    }

    // Prepare a heap with given ys for backprop.
    let mut heap = ys
        .iter()
        .map(|y| y.as_ref().to_node())
        .collect::<BinaryHeap<Node>>();

    // Start backprop from `ys`.
    while let Some(y) = heap.pop() {
        let gxs = {
            let y_grad_info = grad_map.get_mut(y.id);
            let gy = y_grad_info.gradient();

            // Call Op::grad
            let y_tensor = g.tensor(y.id);
            let ctx = GradientContext::new(gy, y_tensor, g);
            let gxs = ctx.compute_input_grads();
            debug_assert_eq!(y_tensor.num_backprop_inputs(), gxs.len());
            gxs
        };

        // Register computed gradients
        let y = g.tensor(y.id);
        for (x, gx) in y.inner().get_backprop_inputs().iter().zip(gxs) {
            let x = x.as_tensor(g);
            let x_grad_info = grad_map.get_mut(x.id);
            if x_grad_info.on_backprop_path {
                if let Some(gx) = gx {
                    let x_not_visited =  x_grad_info.gradients.len() == 0;
                    grad_map.push_grad(x.id, gx);
                    // update heap
                    if !x.is_source() && x_not_visited {
                        heap.push(x.to_node());
                    }
                }
            }
        }
    }

    grad_map
}

// a graph node in a gradient subgraph
struct Node {
    id: usize,
    topo_rank: usize,
}

impl Ord for Node {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.topo_rank.cmp(&other.topo_rank)
    }
}

impl PartialOrd for Node {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.topo_rank.cmp(&other.topo_rank))
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
            topo_rank: self.graph.topo_rank(self.id)
        }
    }
}

pub(crate) struct GradientMap<'graph, F: Float> {
    inner: FxHashMap<TensorID, GradientInfo<'graph, F>>,
}

impl<'graph, F: Float> GradientMap<'graph, F> {
    pub(crate) fn extract_grad(&mut self, x: impl AsRef<Tensor<'graph, F>>) -> Option<Tensor<'graph, F>> {
        if let Some(info) = self.inner.get_mut(&x.as_ref().id) {
            if info.on_backprop_path {
                return Some(info.gradient());
            }
        }
        // can't differentiate!
        None
    }

    #[inline]
    fn get_mut(&mut self, key: TensorID) -> &mut GradientInfo<'graph, F> {
        self.inner.get_mut(&key).unwrap()
    }

    #[inline]
    fn push_grad(&mut self, key: TensorID, grad: Tensor<'graph, F>) {
        self.inner.get_mut(&key).unwrap().gradients.push(grad);
    }
}

// GradientInfo is keyed by a TensorID and holds its gradient info for back-prop
struct GradientInfo<'graph, F: Float> {
    gradients: SmallVec<Tensor<'graph, F>>,
    on_backprop_path: bool,
}

impl<'graph, F: Float> GradientInfo<'graph, F> {
    #[inline]
    fn new(on_backprop_path: bool) -> GradientInfo<'graph, F> {
        GradientInfo {
            on_backprop_path,
            gradients: SmallVec::new(),
        }
    }

    #[inline]
    fn gradient(&mut self) -> Tensor<'graph, F> {
        if self.gradients.len() > 1 { // the accumulated gradients are added together at this time.
            self.gradients[0] = T::add_n(self.gradients.as_slice());
        }
        self.gradients[0]
    }
}

#[inline]
fn has_child_on_path<T: Float>(parent: Tensor<T>, path: &FxHashMap<usize, GradientInfo<T>>) -> bool {
    let inner = parent.inner();
    for child in inner.get_backprop_inputs() {
        if path.get(&child.id).unwrap().on_backprop_path {
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
// Nodes between `ys` and `xs` are marked as `on_backprop_path`.
fn init_gradient_map<'graph, A, B, F: Float>(
    g: &'graph Graph<F>,
    ys: &[A],
    xs: &[B],
) -> GradientMap<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>>,
        B: AsRef<Tensor<'graph, F>>,
{
    let mut map = FxHashMap::<TensorID, GradientInfo<F>>::default();

    // Builds GradientInfo while performing depth-first-search.

    // dfs_stack: (node, should_visit)
    let mut dfs_stack: Vec<(TensorID, bool)> = ys.iter().map(|y| (y.as_ref().id, false)).collect();
    while let Some((curr_id, should_visit)) = dfs_stack.pop() {
        let curr_node = g.tensor(curr_id);
        if should_visit {
            let on_backprop_path =
                curr_node.is_differentiable() && (is_given_xs(curr_id, xs) || has_child_on_path(curr_node, &map));
            map.insert(curr_id, GradientInfo::new(on_backprop_path));
        } else {
            // Put self on the stack top (should visit next time)
            dfs_stack.push((curr_id, true));
            // Push children as necessary
            let curr_node = curr_node.inner();
            for child in curr_node.get_backprop_inputs() {
                let child = child.as_tensor(g);
                if map.get(&curr_id).is_none() {
                    if child.is_source() || !child.is_differentiable() {
                        // Add to result, but don't allow any more recursive search
                        // because there will be no `xs` nodes in this direction....
                        map.insert(
                            child.id,
                            GradientInfo::new(child.is_differentiable() && is_given_xs(child.id, xs)),
                        );
                    } else {
                        // Recurse
                        dfs_stack.push((child.id, false));
                    }
                }
            }
        }
    }
    GradientMap { inner: map }
}