use ndarray;
use ndarray_ext::{NdArray, NdArrayView};
use op;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::rc::Rc;
use tensor::Tensor;
use Float;

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
///     .run(&[(a, &nd::arr0(2.).into_dyn())]);  // Do eval
/// ```
pub struct Eval<'tsr, T: 'tsr + Float> {
    buf: Vec<&'tsr Tensor<T>>,
}

impl<'k, T: Float + 'k> Eval<'k, T> {
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
    pub fn run<F>(&'k self, feed: F) -> Vec<Option<NdArray<T>>>
    where
        F: IntoIterator<Item = Feed<'k, T>>
    {
        eval(&self.buf, feed)
    }
}

// Context in evaluation of `node`
pub struct OpComputeContext<'k, T: Float + 'k> {
    node: &'k Tensor<T>,
    xs: Vec<NdArrayView<'k, T>>,
}

impl<'k, T: Float> OpComputeContext<'k, T> {
    #[inline]
    pub fn new(node: &'k Tensor<T>, xs: Vec<NdArrayView<'k, T>>) -> Self {
        OpComputeContext { node, xs }
    }

    #[inline]
    pub fn get_node(&self) -> &Tensor<T> {
        &self.node
    }

    #[inline]
    pub fn grab_inputs(&self) -> &[NdArrayView<'k, T>] {
        self.xs.as_slice()
    }

    #[inline]
    pub fn grab_input_node(&self, i: usize) -> &Tensor<T> {
        &self.node.inputs[i]
    }

    // Internal of `grab_inputs`.
    // Grabs input arrays for `node`'s evaluation.
    // Returns "None" when failed to aggregate even one of input arrays.
    fn _grab_inputs(
        node: &'k Tensor<T>,
        store: &ResourceStore<'k, T>,
        feed_store: &FeedStore<'k, T>,
    ) -> Option<Vec<(usize, &'k Tensor<T>)>> {
        fn recurse<'k, 'v, T: Float>(
            x: &'k Tensor<T>,
            store: &ResourceStore<'k, T>,
            feed_store: &FeedStore<'v, T>,
            value_index: usize,
        ) -> Option<(usize, &'k Tensor<T>)> {
            if let Some(_) = x.get_persistent_array() {
                Some((value_index, x))
            } else if x.is_placeholder {
                Some((value_index, x))
            } else {
                match store[x.resource_lookup_key.get()].value[value_index] {
                    Ok(_) => Some((value_index, x)),
                    // hoping for x.inputs[i] to have the value
                    Err(::op::ComputeException::Delegate { to: i }) => {
                        recurse(&x.inputs[i], store, feed_store, x.input_indices[i])
                    }
                    _ => None, // None for hopeless errors
                }
            }
        }

        let input_nodes = &node.inputs;
        let mut input_arrays = Vec::with_capacity(input_nodes.len());
        for (x, &i) in input_nodes.into_iter().zip(node.input_indices.iter()) {
            if let Some(res) = recurse(x, store, feed_store, i) {
                input_arrays.push(res);
            } else {
                // Early return
                return None;
            }
        }
        Some(input_arrays)
    }
}

struct NodeWithValue<'a, T: Float + 'a> {
    node: &'a Tensor<T>,
    value: op::ComputeResult<T>,
    // How many resources of this node does user requires.
    // When this is reduced to one, `value` is ready to be moved out (without copy).
    pending_count: usize,
}

impl<'a, T: Float> Tensor<T> {
    #[inline]
    fn with_value(&'a self, val: op::ComputeResult<T>) -> NodeWithValue<'a, T> {
        NodeWithValue {
            node: self,
            value: val,
            pending_count: 0,
        }
    }
}

pub struct Feed<'k, T: Float>(&'k Tensor<T>, ndarray::ArrayView<'k, T, ndarray::IxDyn>);

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
// FIXME: Annoying lifetime params
pub fn eval<'c, 'k, V, U, T: Float + 'k>(
    tensors: &'c [V],  // output_storage などは
    feeds: U,
) -> Vec<Option<NdArray<T>>>
where
    V: AsRef<Tensor<T>>,
    U: IntoIterator<Item = Feed<'k, T>>,
{
    // Run graph
    let feeds = feeds.into_iter().collect::<Vec<_>>();

    let mut output_storage = eval_internal(
        tensors.iter().map(|t| t.as_ref()).collect::<Vec<_>>().as_slice(), feeds.as_slice());

    // Treat in-place or delegation ops
    let creators = tensors
        .iter()
        .map(|x| {
            let x = x.as_ref();
            let creator = if x.is_placeholder || x.has_persistent_array() {
                x
            } else {
                let creator = find_resource_creator(&output_storage, x);
                output_storage[creator.resource_lookup_key.get()].pending_count += 1;
                creator
            };
            creator
        })
        .collect::<Vec<&Tensor<T>>>();

    // Shrink to fit (output_storage is moved)
    let mut key2res: BTreeMap<usize, NodeWithValue<T>> = finalize_resource_store(output_storage);

    // Aggregate return values
    creators
        .iter()
        .map(|ref creator| {
            if let Some(per) = creator.get_persistent_array() {
                // Rarely happens (case that a persistent array given by user is required)
                Some(per.clone())
            } else if creator.is_placeholder {
                // Rarely happens (case that a feed by user is required)
                Some(find_fed_resource(creator, &feeds).to_owned())
            } else {
                let res = match key2res.entry(creator.resource_lookup_key.get()) {
                    Entry::Occupied(mut ent) => {
                        if ent.get().pending_count == 1 {
                            // move out the resource.
                            let mut got = ent.remove();
                            // Using first item of the compute result
                            map_err(got.value.remove(0))
                        } else {
                            // "clone" the resource.
                            let mut got = ent.get_mut();
                            got.pending_count -= 1;
                            // Using first item of the compute result
                            map_err(got.value[0].clone())
                        }
                    }
                    _ => unreachable!(),
                };
                res
            }
        })
        .collect()
}

// Recursive function which seeks a node holding the x's resource.
// Actual recursion "rarely" happens.
fn find_resource_creator<'a, T: Float>(
    storage: &ResourceStore<T>,
    x: &'a Tensor<T>,
) -> &'a Tensor<T> {
    match storage[x.resource_lookup_key.get()].value[0] {
        Err(::op::ComputeException::Delegate { to: i }) => {
            find_resource_creator(storage, &x.inputs[i])
        }
        _ => x,
    }
}

#[inline]
fn map_err<'a, T: Float>(res: Result<NdArray<T>, ::op::ComputeException>) -> Option<NdArray<T>> {
    match res {
        Ok(arr) => Some(arr),
        Err(::op::ComputeException::NoOutput) => None,
        _ => unreachable!(),
    }
}

type ResourceStore<'a, T> = Vec<NodeWithValue<'a, T>>;
type FeedStore<'a, T> = Vec<NdArrayView<'a, T>>;

#[inline]
fn find_fed_resource<'c, 'k: 'c, T: Float>(
    node: &'k Tensor<T>,
    feeds: &'c [Feed<'k, T>],
) -> NdArrayView<'c, T> {
    // Linear search is suitable because the number of feeds are so small in most cases.
    for feed in feeds {
        if Rc::ptr_eq(feed.0, node) {
            return feed.1.view();
        }
    }
    panic!("Placeholder unfilled.");
}

// Evaluates "targets".
fn eval_internal<'k, 'c, T: Float>(
    targets: &'c [&'k Tensor<T>],
    feeds: &[Feed<'k, T>],
) -> ResourceStore<'k, T>
{
    let mut res_store: Vec<NodeWithValue<'k, T>> = Vec::new();
    let mut feed_store = Vec::new();

    // Obtain array resources while visiting nodes in topological order.
    // Stack-based depth-first-search is used to avoid stack overflow in explicit recursion.
    let mut dfs_stack: Vec<(&Tensor<T>, bool)> = targets.iter().map(|&x| (x, false)).collect();
    while let Some((node, is_parent)) = dfs_stack.pop() {
        if is_parent {
            // Visit this node
            if node.is_placeholder {
                node.resource_lookup_key.set(feed_store.len());
                // ここで return してる
                feed_store.push(find_fed_resource(node, feeds));
            } else {
                node.resource_lookup_key.set(res_store.len());
                if !node.has_persistent_array() {
                    let y = {
                        let in_nodes = OpComputeContext::_grab_inputs(node, &res_store, &feed_store);
                        if let Some(in_nodes_) = in_nodes {
                            let mut xs = Vec::with_capacity(in_nodes_.len());
                            for (value_index, x) in in_nodes_ {
                                if let Some(per) = x.get_persistent_array() {
                                    xs.push(per.view());
                                } else if x.is_placeholder {
                                    xs.push(feed_store[x.resource_lookup_key.get()].view())
                                } else {
                                    let tmp = res_store[x.resource_lookup_key.get()].value[value_index].as_ref().unwrap().view();
                                    xs.push(tmp) // ノードのライフタイム
                                }

                            }
                            node.op.compute(OpComputeContext { node, xs })
                        } else {
                            vec![Err(::op::ComputeException::Delegate { to: 0 })]
                        }
                    };
                    res_store.push(node.with_value(y));
                }
            }
        } else {
            // Update dfs stack
            dfs_stack.push((node, true));
            // Push children if needed
            for child in &node.inputs {
                let visited = {
                    let k = child.resource_lookup_key.get();
                    k < res_store.len() && Rc::ptr_eq(child, res_store[k].node)
                };
                if !visited {
                    dfs_stack.push((child, false));
                }
            }
        }

    }
    res_store
}

// Shrink it by dropping useless resources
// and convert it into mappings of {lookup key => resource}
fn finalize_resource_store<T: Float>(
    mut vec: ResourceStore<T>,
) -> BTreeMap<usize, NodeWithValue<T>> {
    let mut retained_keys = Vec::new();
    let len = vec.len();
    {
        let mut_ref = &mut vec;
        let mut del = 0;
        // Align the resources. Unused resources are placed in the latter part.
        {
            let v = &mut **mut_ref;

            for i in 0..len {
                if v[i].pending_count == 0 {
                    del += 1;
                    continue;
                }
                retained_keys.push(i);
                if del > 0 {
                    // slides i th node forward
                    v.swap(i - del, i);
                }
            }
        }
        if del > 0 {
            // Drop unused resources here
            mut_ref.truncate(len - del);
        }
    }
    debug_assert_eq!(vec.len(), retained_keys.len());
    // `retained_keys` are sorted automatically.
    // `vec` is moved into Map.
    retained_keys.into_iter().zip(vec).collect()
}

#[test]
fn test_eval() {
    let ref v = ::ops::placeholder::<f32>(&[3, 2, 1]);
    let ref z = ::ops::reduce_sum(&::ops::squeeze(v, &[2]), &[0, 1], false);
    let ref g = ::ops::grad(&[z], &[v]);
    let eval_result = &eval(g, &[(v, &::ndarray_ext::ones(&[3, 2, 1]))])[0];
    assert_eq!(eval_result.as_ref().unwrap().shape(), &[3, 2, 1]);
}

#[test]
fn test_constant_eval() {
    let arr = ndarray::arr1(&[0., 0., 0.]);
    assert_eq!(Some(arr.clone().into_dyn()), ::variable(arr).eval(&[]));
}

#[test]
fn test_placeholder_eval() {
    let arr = ::ndarray_ext::ones::<f32>(&[3, 2, 1]);
    let ref v = ::ops::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[(v, &arr)]);
    assert_eq!(eval_result[0], Some(arr));
}

#[test]
fn test_eval_internal() {
    let ref v = ::ops::placeholder::<f32>(&[3, 2, 1]);
    let ref z = ::ops::squeeze(v, &[2]);
    let ref g = ::ops::grad_with_default(&[z], &[v], &[&::ones(&z.shape())]);
    let storage = eval_internal(&vec![&g[0]], &vec![&(v, &::ndarray_ext::ones(&[3, 2, 1]))]);

    assert_eq!(
        storage.iter().map(|x| x.node.op.name()).collect::<Vec<_>>(),
        vec![
            "ConvertToTensor",
            "Squeeze", // forward end
            "Shape",
            "Ones",
            "ExpandDims",
        ]
    );
}
