use crate::ndarray_ext::{NdArray, NdArrayView};
use crate::op;
use crate::tensor::Tensor;
use crate::Float;
use ndarray;
use std::mem;
use std::rc::Rc;

/// Helper structure for batched evaluation.
///
/// Use this in case `ag::eval` doesn't help.
///
/// ```
/// extern crate autograd as ag;
/// extern crate ndarray as nd;
///
/// let ref a = ag::placeholder(&[]);
/// let ref x = a + a;
/// let ref y = a * a;
/// let ref z = a / a;
///
/// ag::Eval::new()
///     .push(&y)
///     .extend(&[y, z])
///     .run(&[ag::Feed(a, nd::arr0(2.).into_dyn().view())]);  // Do eval
/// ```
pub struct Eval<'k, T: Float> {
    buf: Vec<&'k Tensor<T>>,
}

impl<'c, 'k, 'v, T: Float> Eval<'k, T> {
    /// Instantiates a new evaluation session.
    pub fn new() -> Self {
        Eval { buf: Vec::new() }
    }

    /// Appends a tensor to the back of the evaluation targets.
    pub fn push(&mut self, x: &'k Tensor<T>) -> &mut Self {
        self.buf.push(x);
        self
    }

    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'k [A]) -> &mut Self
    where
        A: AsRef<Tensor<T>>,
    {
        self.buf.extend(xs.iter().map(|x| x.as_ref()));
        self
    }

    /// Evaluates the buffered tensors.
    ///
    /// `feeds` is a stream of `(placeholder tensor, its value)`
    pub fn run(&'k self, feeds: &'c [crate::runtime::Feed<'k, 'v, T>]) -> Vec<Option<NdArray<T>>> {
        eval(&self.buf, feeds)
    }
}

// Context in evaluation of `node`
pub struct OpComputeContext<'v, T: Float> {
    nodes: Vec<Tensor<T>>,
    xs: Vec<NdArrayView<'v, T>>,
}

impl<'v, T: Float> OpComputeContext<'v, T> {
    #[inline]
    pub fn new(nodes: Vec<Tensor<T>>, xs: Vec<NdArrayView<'v, T>>) -> Self {
        OpComputeContext { nodes, xs }
    }

    #[inline]
    pub fn node(&self, idx: usize) -> &Tensor<T> {
        &self.nodes[idx]
    }

    #[inline]
    pub fn grab_inputs(&self) -> &[NdArrayView<'v, T>] {
        self.xs.as_slice()
    }
}

#[derive(Clone, Copy)]
enum ValueType {
    Owned,
    View,
    NoOutput,
}

struct ValueInfo<T: Float> {
    ty: ValueType,
    // key to lookup the value
    key: usize,
    // Owned array
    value: Option<NdArray<T>>,
}

impl<T: Float> ValueInfo<T> {
    #[inline]
    fn new(ty: ValueType, key: usize) -> ValueInfo<T> {
        ValueInfo {
            ty,
            key,
            value: None,
        }
    }
}

struct NodeWithValueInfo<'k, T: Float> {
    node: &'k Tensor<T>,
    info_list: Vec<ValueInfo<T>>,
    contains_no_output: bool,
}

impl<'k, T: Float> Tensor<T> {
    #[inline]
    fn with_value_info(
        &'k self,
        info_list: Vec<ValueInfo<T>>,
        contains_no_output: bool,
    ) -> NodeWithValueInfo<'k, T> {
        NodeWithValueInfo {
            node: self,
            info_list,
            contains_no_output,
        }
    }
}

pub struct Feed<'k, 'f, T: Float>(
    pub &'k Tensor<T>,                             // a placeholder tensor
    pub ndarray::ArrayView<'f, T, ndarray::IxDyn>, // its value
);

/// Evaluates given symbolic tensors.
///
/// Each return value can be `None`;
/// for example, evaluation of `gradient_descent_ops::*`
/// would result in `None`.
///
/// NOTE: All the runtime errors are not reported by return values, but by "panic"
/// for convenience.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[2]);
/// let ref b = ag::ones(&[2]);
///
/// // eval two tensors at once.
/// let evaluated = ag::eval(&[a, b], &[]);
/// assert_eq!(evaluated[0], Some(ndarray::arr1(&[0., 0.]).into_dyn()));
/// assert_eq!(evaluated[1], Some(ndarray::arr1(&[1., 1.]).into_dyn()));
/// ```
#[allow(mutable_transmutes, unused_mut)]
pub fn eval<'slice, 'node, 'feed, K, T>(
    tensors: &'node [K],
    feeds: &'slice [Feed<'node, 'feed, T>],
) -> Vec<Option<NdArray<T>>>
where
    K: AsRef<Tensor<T>>,
    T: Float,
{
    let mut node_info_storage: Vec<NodeWithValueInfo<T>> = Vec::new();
    let mut owned_storage: Vec<NdArray<T>> = Vec::new();

    {
        let mut view_storage: Vec<NdArrayView<T>> = Vec::new();
        let mut feed_store: Vec<NdArrayView<'feed, T>> = Vec::new();

        let mut dfs_stack: Vec<(&Tensor<T>, bool)> =
            tensors.iter().map(|x| (x.as_ref(), false)).collect();

        // Obtain array resources while visiting nodes in topological order.
        // Stack-based depth-first-search is used to avoid stack overflow in explicit recursion.
        while let Some((node, is_parent)) = dfs_stack.pop() {
            if is_parent {
                // Visit this node
                if node.is_placeholder {
                    node.resource_lookup_key.set(feed_store.len());
                    let mut found = None;
                    for feed in feeds {
                        if Rc::ptr_eq(feed.0, node) {
                            found = Some(feed.1.clone());
                            break;
                        }
                    }
                    unsafe {
                        mem::transmute::<_, &mut Vec<_>>(&feed_store)
                            .push(found.expect("Placeholder unfilled."))
                    }
                } else {
                    node.resource_lookup_key.set(node_info_storage.len());
                    if !node.has_persistent_array() {
                        // Aggregate input arrays
                        let mut err = None;
                        let mut xs = Vec::with_capacity(node.inputs.len());
                        for (x, &i) in node.inputs.iter().zip(&node.input_indices) {
                            if let Some(per) = x.get_persistent_array() {
                                xs.push(per.view());
                            } else if x.is_placeholder {
                                xs.push(feed_store[x.resource_lookup_key.get()].view());
                            } else {
                                // Require computed outputs
                                let k = x.resource_lookup_key.get();
                                if node_info_storage[k].contains_no_output {
                                    err = Some(vec![Err(op::ComputeException::NoOutput)]);
                                    break;
                                } else {
                                    let info = &node_info_storage[k].info_list[i];
                                    match info.ty {
                                        ValueType::Owned => {
                                            xs.push(owned_storage[info.key].view());
                                        }
                                        ValueType::View => {
                                            // Clone the view
                                            xs.push(view_storage[info.key].clone());
                                        }
                                        ValueType::NoOutput => {
                                            err = Some(vec![Err(op::ComputeException::NoOutput)]);
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        // Call Op::compute
                        let ys: op::ComputeResults<_> = match err {
                            Some(e) => e,
                            _ => node.op.compute(OpComputeContext {
                                nodes: node.inputs.clone(),
                                xs,
                            }),
                        };
                        // Aggregate compute result
                        let mut info_list = Vec::with_capacity(ys.len());
                        let mut contains_no_output = false;
                        for y in ys {
                            match y {
                                Ok(crate::ArrRepr::Owned(val)) => {
                                    info_list.push(ValueInfo::new(
                                        ValueType::Owned,
                                        owned_storage.len(),
                                    ));
                                    unsafe {
                                        // safe
                                        mem::transmute::<_, &mut Vec<_>>(&owned_storage).push(val);
                                    }
                                }
                                Ok(crate::ArrRepr::View(val)) => {
                                    info_list
                                        .push(ValueInfo::new(ValueType::View, view_storage.len()));
                                    view_storage.push(val);
                                }
                                _ => {
                                    info_list.push(ValueInfo::new(ValueType::NoOutput, 0));
                                    contains_no_output = true;
                                }
                            }
                        }
                        node_info_storage.push(node.with_value_info(info_list, contains_no_output))
                    };
                }
            } else {
                // Update dfs stack
                dfs_stack.push((node, true));
                // Push children if needed
                for child in &node.inputs {
                    if !visited(child, &node_info_storage) {
                        dfs_stack.push((child, false));
                    }
                }
            }
        } // while loop end

        // process array views
        for t in tensors {
            let t = t.as_ref();
            if !t.is_placeholder && !t.has_persistent_array() {
                let info = &node_info_storage[t.resource_lookup_key.get()].info_list[0];
                if let ValueType::View = info.ty {
                    node_info_storage[t.resource_lookup_key.get()].info_list[0].value =
                        Some(view_storage[info.key].to_owned());
                }
            }
        }
    } // lifetime of views ends here

    for t in tensors {
        let t = t.as_ref();
        if !t.is_placeholder && !t.has_persistent_array() {
            let info = &node_info_storage[t.resource_lookup_key.get()].info_list[0];
            if let ValueType::Owned = info.ty {
                node_info_storage[t.resource_lookup_key.get()].info_list[0].value =
                    Some(owned_storage[info.key].to_owned());
            }
        }
    }

    let mut ret: Vec<Option<NdArray<T>>> = Vec::with_capacity(tensors.len());
    for t in tensors {
        let t = t.as_ref();
        let arr = if let Some(per) = t.get_persistent_array() {
            // rare case
            Some(per.clone())
        } else if t.is_placeholder {
            // rare case
            let mut found = None;
            for feed in feeds {
                if Rc::ptr_eq(feed.0, t) {
                    found = Some(&feed.1);
                    break;
                }
            }
            Some(found.expect("Placeholder unfilled.").to_owned())
        } else {
            mem::replace(
                &mut node_info_storage[t.resource_lookup_key.get()].info_list[0].value,
                None,
            )
        };
        ret.push(arr);
    }
    ret
}

#[inline(always)]
fn visited<T: Float>(node: &Tensor<T>, node_info_storage: &Vec<NodeWithValueInfo<T>>) -> bool {
    let k = node.resource_lookup_key.get();
    k < node_info_storage.len() && Rc::ptr_eq(node, node_info_storage[k].node)
}

#[test]
fn test_eval() {
    let ref v = crate::ops::placeholder::<f32>(&[3, 2, 1]);
    let ref z = crate::ops::reduce_sum(&crate::ops::squeeze(v, &[2]), &[0, 1], false);
    let ref g = crate::ops::grad(&[z], &[v]);
    let eval_result = &eval(g, &[Feed(v, crate::ndarray_ext::ones(&[3, 2, 1]).view())])[0];
    assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
}

#[test]
fn test_constant_eval() {
    let arr = ndarray::arr1(&[0., 0., 0.]);
    assert_eq!(Some(arr.clone().into_dyn()), crate::variable(arr).eval(&[]));
}

#[test]
fn test_placeholder_eval() {
    let arr = crate::ndarray_ext::ones::<f32>(&[3, 2, 1]);
    let ref v = crate::ops::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[Feed(v, arr.view())]);
    assert_eq!(eval_result[0], Some(arr));
}
