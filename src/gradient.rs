use ops;
use std::cmp::Ordering;
use std::collections::binary_heap::BinaryHeap;
use std::fmt;
use std::mem;
use std::rc::Rc;
use tensor::Tensor;
use Float;

struct GradInfo<'a, T: Float + 'a> {
    node: &'a Tensor<T>, // information of this node
    has_gradient: bool,
    grad_called: bool,
    computed_grads: Vec<Tensor<T>>,
    default_grad: Option<&'a Tensor<T>>,
}

impl<'a, T: Float> GradInfo<'a, T> {
    #[inline]
    fn new(
        t: &'a Tensor<T>,
        has_gradient: bool,
        default_grad: Option<&'a Tensor<T>>,
    ) -> GradInfo<'a, T> {
        GradInfo {
            node: t,
            has_gradient,
            computed_grads: Vec::new(),
            grad_called: false,
            default_grad,
        }
    }
}

impl<'a, T: Float> fmt::Debug for GradInfo<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.node.op.name())
    }
}

impl<T: Float> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.op.name())
    }
}

macro_rules! access_grad_info_of {
    ($node:expr, $path:expr) => {
        $path[$node.resource_lookup_key.get()]
    };
}

#[inline]
fn has_marked_child<T: Float>(parent: &Tensor<T>, path: &Vec<GradInfo<T>>) -> bool {
    let mut it = if let Some(ref a) = parent.inputs_on_backprop {
        a.iter()
    } else {
        parent.inputs.iter()
    };
    while let Some(child) = it.next() {
        if access_grad_info_of!(child, path).has_gradient {
            return true;
        }
    }
    false
}

// Marks `has_gradient` if each node is on the gradient propagation path.
//
// Strategy
//   1. Record all nodes that are reachable from `ys` into `path`.
//   2. Mark the path between `ys` and `xs` as `has_gradient`.
fn mark_gradient_path<'a, T: Float>(
    ys: &[&'a Tensor<T>],
    xs: &[&'a Tensor<T>],
) -> Vec<GradInfo<'a, T>> {
    // Randomly accessible by use of each node's lookup key.
    let mut path: Vec<GradInfo<'a, T>> = Vec::new();

    // Builds GradInfo while performing depth-first-search.
    // `has_gradient` properties are filled at the same time.
    let mut dfs_stack: Vec<(&Tensor<T>, bool)> = ys.iter().map(|&y| (y, false)).collect();
    while let Some((node, should_visit)) = dfs_stack.pop() {
        if should_visit {
            let marker = xs.contains(&node) || has_marked_child(node, &path);
            node.resource_lookup_key.set(path.len());
            path.push(GradInfo::new(node, marker, None));
        } else {
            dfs_stack.push((node, true));
            // Push children as necessary
            let children = if let Some(ref a) = node.inputs_on_backprop {
                a
            } else {
                &node.inputs
            };
            for child in children {
                let visited = {
                    let k = child.resource_lookup_key.get();
                    k < path.len() && Rc::ptr_eq(child, path[k].node)
                };
                if !visited {
                    if child.is_source() || !child.is_differentiable {
                        // Add to result, but don't allow any more recursive search
                        child.resource_lookup_key.set(path.len());
                        path.push(GradInfo::new(child, xs.contains(&child), None));
                    } else {
                        // Recurse
                        dfs_stack.push((child, false));
                    }
                }
            }
        }
    }
    path
}

#[test]
fn test_gradient_path() {
    // dummy graph
    // y = 3 * x1 * x1 + 5 * x2 + x3;
    let ref x1 = ::ops::placeholder(&[]);
    let ref x2 = ::ops::placeholder(&[]);
    let ref x3 = ::ops::placeholder(&[]);
    let ref a = 3. * x1; // rank 1
    let ref b = a * x1; // rank 2
    let ref c = 5. * x2; // rank 1
    let ref d = b + c; // rank 3
    let ref y = d + x3; // rank 4
    let path = mark_gradient_path(&[y], &[x1, x2]);
    let path_: Vec<&Tensor<f64>> = path.iter().map(|a| a.node).collect();

    assert!(path_.contains(&x1));
    assert!(path_.contains(&x2));
    assert!(path_.contains(&x3));
    assert!(path_.contains(&a));
    assert!(path_.contains(&b));
    assert!(path_.contains(&c));
    assert!(path_.contains(&d));
    assert!(path_.contains(&y));
    assert_eq!(path_.len(), 10); // number of nodes in the grad path

    // Topological ordering test
    let ix1 = path_.iter().position(|x| Rc::ptr_eq(x, x1)).unwrap();
    let ix2 = path_.iter().position(|x| Rc::ptr_eq(x, x2)).unwrap();
    let ix3 = path_.iter().position(|x| Rc::ptr_eq(x, x3)).unwrap();
    let ia = path_.iter().position(|x| Rc::ptr_eq(x, a)).unwrap();
    let ic = path_.iter().position(|x| Rc::ptr_eq(x, c)).unwrap();
    let ib = path_.iter().position(|x| Rc::ptr_eq(x, b)).unwrap();
    let id = path_.iter().position(|x| Rc::ptr_eq(x, d)).unwrap();
    let iy = path_.iter().position(|x| Rc::ptr_eq(x, y)).unwrap();
    assert!(ix1 < ia);
    assert!(ix2 < ic);
    assert!(ix3 < iy);
    assert!(ib < id);
    assert!(id < iy);

    // Ensure continuity of keys
    for (i, node) in path_.iter().enumerate() {
        assert_eq!(i, node.resource_lookup_key.get());
    }

    // Connection test
    use std::collections::btree_set::BTreeSet;
    let all = (0..10).collect::<BTreeSet<usize>>();
    let should_be_has_gradient = [ix1, ix2, ia, ic, ib, id, iy]
        .into_iter()
        .cloned()
        .collect::<BTreeSet<usize>>();
    for &id in should_be_has_gradient.iter() {
        if !path[id].has_gradient {
            panic!("{} is not has_gradient", path[id].node.op.name());
        }
    }
    for &id in all.difference(&should_be_has_gradient).into_iter() {
        if path[id].has_gradient {
            panic!("{} should not be has_gradient", path[id].node.op.name());
        }
    }
}

/// Returns symbolic gradient tensors of `xs`.
///
/// This computes partial derivatives of `ys` with `xs` and returns the
/// gradients. This is achieved by building a subgraph between `ys` and
/// `xs` in reverse order from user's graph definition.
/// `known_gys` are already known gradients of `ys`'s outputs.
///
/// NOTE: Nodes that do not have gradients won't be included in the subgraph to avoid
/// unnecessary computation.
pub fn symbolic_gradients<T: Float>(
    ys: &[&Tensor<T>],
    xs: &[&Tensor<T>],
    known_gys: &[Option<&Tensor<T>>],
) -> Vec<Tensor<T>> {
    assert_eq!(
        ys.len(),
        known_gys.len(),
        "`ys.len()` must match `gys.len()`"
    );

    // Setup gradient path.
    let mut path = mark_gradient_path(ys, xs);

    // Set default grads.
    for (y, gy) in ys.iter().zip(known_gys) {
        let y_info = &mut access_grad_info_of!(y, path);
        if let &Some(gy_) = gy {
            y_info.default_grad = Some(gy_);
        } else {
            y_info.computed_grads.push(ops::scalar(T::one()));
        }
    }

    // Prepare a heap with given ys.
    let mut heap = ys
        .into_iter()
        .map(|y| y.wrapped())
        .collect::<BinaryHeap<TensorWrapper<T>>>();

    // Backprop.
    // c.f. https://github.com/chainer/chainer/blob/master/chainer/variable.py
    while let Some(y) = heap.pop() {
        let xs_ = y
            .inner
            .inputs
            .iter()
            .map(|a| a)
            .collect::<Vec<&Tensor<T>>>();
        let gxs = {
            let info = &mut access_grad_info_of!(y.inner, path);
            let gy = if let Some(def) = info.default_grad {
                def
            } else {
                let gys: &mut Vec<_> = &mut info.computed_grads;
                accumulate_grads_if_needed(gys);
                &gys[0]
            };
            // Call Op::grad
            let gxs = y.inner.op.grad(gy, xs_.as_slice(), y.inner);
            // Validate y::op::grad implementation
            debug_assert_eq!(
                xs_.len(),
                gxs.len(),
                "{}::grad returned {} gxs",
                y.inner,
                gxs.len()
            );
            gxs
        };
        // Register computed gradients
        let xs_ = if let Some(ref a) = y.inner.inputs_on_backprop {
            a
        } else {
            &y.inner.inputs
        };
        for (gx, x) in gxs.into_iter().zip(xs_) {
            let x_info = &mut access_grad_info_of!(x, path);
            if x_info.has_gradient {
                if let Some(gx_) = gx {
                    x_info.computed_grads.push(gx_);
                    // update heap
                    if !x.is_source() && !x_info.grad_called {
                        x_info.grad_called = true;
                        heap.push(x.wrapped());
                    }
                }
            }
        }
    }

    // Aggregate and return xs's gradients
    xs.iter()
        .map(|x| {
            let xk = x.resource_lookup_key.get();
            assert!(
                xk < path.len() && Rc::ptr_eq(x, path[xk].node),
                "Not differentiable with given tensor(s)."
            );
            let info = &mut path[xk];
            assert!(
                info.default_grad.is_none(),
                "Can't differentiate with objective itself"
            );
            let gxs = &mut info.computed_grads;
            accumulate_grads_if_needed(gxs);
            debug_assert_eq!(gxs.len(), 1);
            gxs.remove(0)
        })
        .collect::<Vec<Tensor<T>>>()
}

struct TensorWrapper<'a, T: Float + 'a> {
    inner: &'a Tensor<T>,
}

impl<'a, T: Float> Ord for TensorWrapper<'a, T> {
    // Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering {
        self.inner.top_rank.cmp(&other.inner.top_rank)
    }
}

impl<'a, T: Float> PartialOrd for TensorWrapper<'a, T> {
    #[inline]
    // Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.inner.top_rank.cmp(&other.inner.top_rank))
    }
}

impl<'a, T: Float> Eq for TensorWrapper<'a, T> {}

impl<'a, T: Float> PartialEq for TensorWrapper<'a, T> {
    #[inline]
    fn eq(&self, other: &TensorWrapper<'a, T>) -> bool {
        Rc::ptr_eq(&self.inner, &other.inner)
    }
}

impl<T: Float> Tensor<T> {
    #[inline]
    fn wrapped(&self) -> TensorWrapper<T> {
        TensorWrapper { inner: self }
    }
}

#[inline]
fn accumulate_grads_if_needed<T: Float>(grads: &mut Vec<Tensor<T>>) {
    if grads.len() > 1 {
        let mut acc = {
            let refs = grads.iter().map(|a| a).collect::<Vec<_>>();
            ::ops::add_n(refs.as_slice())
        };
        mem::swap(&mut acc, &mut grads[0]);
        grads.truncate(1)
    }
}
