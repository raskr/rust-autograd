use ndarray;
use ndarray_ext::NdArray;
use op;
use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::mem;
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

impl<'tpl, 'tsr: 'tpl, 'arr: 'tpl, T: Float + 'tsr + 'arr> Eval<'tsr, T> {
    /// Instantiates a new evaluation session.
    pub fn new() -> Self {
        Eval { buf: Vec::new() }
    }

    /// Appends a tensor to the back of the evaluation targets.
    pub fn push(&mut self, x: &'tsr Tensor<T>) -> &mut Self {
        self.buf.push(x);
        self
    }

    /// Extends the evaluation targets with `xs`.
    pub fn extend<A>(&mut self, xs: &'tsr [A]) -> &mut Self
    where
        A: AsRef<Tensor<T>>,
    {
        self.buf.extend(xs.iter().map(|x| x.as_ref()));
        self
    }

    /// Evaluates the buffered tensors.
    ///
    /// `feeds` is a stream of `(placeholder tensor, its value)`
    pub fn run<F>(&self, feed: F) -> Vec<Option<NdArray<T>>>
    where
        F: IntoIterator<Item = &'tpl (&'tsr Tensor<T>, &'arr ndarray::Array<T, ndarray::IxDyn>)>,
    {
        eval(&self.buf, feed)
    }
}

// Context in evaluation of `node`
pub struct OpComputeContext<'a, 'b, T: Float + 'a + 'b> {
    node: &'a Tensor<T>,
    xs: Vec<&'b NdArray<T>>,
}

impl<'a, 'b, T: Float> OpComputeContext<'a, 'b, T> {
    #[inline]
    pub fn new(node: &'a Tensor<T>, xs: Vec<&'b NdArray<T>>) -> Self {
        OpComputeContext { node, xs }
    }

    #[inline]
    pub fn get_node(&self) -> &Tensor<T> {
        &self.node
    }

    #[inline]
    pub fn grab_inputs(&self) -> &[&NdArray<T>] {
        self.xs.as_slice()
    }

    #[inline]
    #[allow(mutable_transmutes)]
    pub unsafe fn grab_assignable_inputs(&mut self) -> &mut [&mut NdArray<T>] {
        mem::transmute(self.xs.as_slice())
    }

    #[inline]
    pub fn grab_input_node(&self, i: usize) -> &Tensor<T> {
        &self.node.inputs[i]
    }

    // Internal of `grab_inputs`.
    // Grabs input arrays for `node`'s evaluation.
    // Returns "None" when failed to aggregate even one of input arrays.
    fn _grab_inputs<'n, 's: 'n>(
        node: &'s Tensor<T>,
        store: &'n ResourceStore<T>,
        feed_store: &FeedStore<'n, T>,
    ) -> Option<Vec<&'n NdArray<T>>> {
        fn recurse<'n, 's: 'n, T: Float>(
            x: &'s Tensor<T>,
            store: &'n ResourceStore<T>,
            feed_store: &FeedStore<'n, T>,
            value_index: usize,
        ) -> Option<&'n NdArray<T>> {
            if let Some(per) = x.get_persistent_array() {
                Some(per)
            } else if x.is_placeholder {
                Some(feed_store[x.resource_lookup_key.get()])
            } else {
                match store[x.resource_lookup_key.get()].value[value_index] {
                    Ok(ref a) => Some(a),
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
pub fn eval<'a, 'b: 'a, 'c: 'a, V, U, T: Float + 'c + 'b>(
    tensors: &[V],
    feeds: U,
) -> Vec<Option<NdArray<T>>>
where
    V: AsRef<Tensor<T>>,
    U: IntoIterator<Item = &'a (&'b Tensor<T>, &'c ndarray::Array<T, ndarray::IxDyn>)>,
{
    // Run graph
    let feeds = feeds.into_iter().collect::<Vec<_>>();
    let mut output_storage = eval_internal(&tensors.iter().map(|t| t.as_ref()).collect(), &feeds);

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
                Some(find_fed_resource(creator, &feeds).clone())
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
fn find_resource_creator<'a, 'b, T: Float>(
    storage: &ResourceStore<T>,
    x: &'b Tensor<T>,
) -> &'b Tensor<T> {
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
type FeedStore<'a, T> = Vec<&'a NdArray<T>>;

#[inline]
fn find_fed_resource<'a, T: Float>(
    node: &Tensor<T>,
    feeds: &Vec<&(&Tensor<T>, &'a NdArray<T>)>,
) -> &'a NdArray<T> {
    // Linear search is suitable because the number of feeds are so small in most cases.
    for feed in feeds {
        if Rc::ptr_eq(feed.0, node) {
            return feed.1;
        }
    }
    panic!("Placeholder unfilled.");
}

// Evaluates "targets".
fn eval_internal<'a, T: Float>(
    targets: &Vec<&'a Tensor<T>>,
    feeds: &Vec<&(&'a Tensor<T>, &NdArray<T>)>,
) -> ResourceStore<'a, T> {
    let mut res_store = Vec::new();
    let mut feed_store = Vec::new();

    // Obtain array resources while visiting nodes in topological order.
    // Stack-based depth-first-search is used to avoid stack overflow in explicit recursion.
    let mut dfs_stack: Vec<(&Tensor<T>, bool)> = targets.iter().map(|&x| (x, false)).collect();
    while let Some((node, is_parent)) = dfs_stack.pop() {
        if is_parent {
            // Visit this node
            if node.is_placeholder {
                node.resource_lookup_key.set(feed_store.len());
                feed_store.push(find_fed_resource(node, &feeds));
            } else {
                node.resource_lookup_key.set(res_store.len());
                if !node.has_persistent_array() {
                    let y = {
                        let ins = OpComputeContext::_grab_inputs(node, &res_store, &feed_store);
                        if let Some(xs) = ins {
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
