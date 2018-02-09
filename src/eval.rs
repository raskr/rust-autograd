//! Implements graph evaluation algorithms.
extern crate ndarray;

use ndarray_ext::NdArray;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::mem;
use std::rc::Rc;
use tensor::Tensor;


pub struct OpComputeContext<'a, 'b: 'a> {
    pub node: &'b Tensor, // Expose to its op for convenience
    resource_store: &'a ResourceStore<'b>,
    feed_store: &'a FeedStore<'b>,
}

impl<'a, 'b> OpComputeContext<'a, 'b> {

    #[inline]
    fn new(
        node: &'b Tensor,
        resource_store: &'a ResourceStore<'b>,
        feed_store: &'a FeedStore<'b>,
    ) -> Self
    {
        OpComputeContext { node, resource_store, feed_store }
    }

    #[inline]
    pub fn grab_inputs(&self) -> Vec<&NdArray>
    {
        self.grab_inputs_internal(false)
    }

    #[inline]
    pub unsafe fn grab_assignable_inputs(&self) -> Vec<&mut NdArray>
    {
        mem::transmute(self.grab_inputs_internal(true))
    }

    #[inline]
    fn grab_inputs_internal(&self, mutable: bool) -> Vec<&NdArray>
    {
        fn seek<'a, 'b: 'a>(
            x: &'b Tensor,
            store: &'a ResourceStore,
            feed_store: &FeedStore<'a>,
            mutable: bool,
        ) -> &'a NdArray
        {
            if let Some(ref per) = x.persistent_array {
                assert!(!mutable || x.op.name() != "Const");
                per
            } else if x.is_placeholder {
                feed_store[x.resource_lookup_key.get()]
            } else {
                match store[x.resource_lookup_key.get()].value {
                    Ok(ref res) => res,
                    // hoping for x.inputs[i] to have the value
                    Err(::OpComputeErrorStatus::Delegate { to: i }) => {
                        seek(&x.inputs[i], store, feed_store, mutable)
                    }
                    Err(::OpComputeErrorStatus::NoOutput) => {
                        panic!("autograd failed: {}'s output not usable", x)
                    }
                    // panic
                    Err(::OpComputeErrorStatus::BadInput(ref msg)) => {
                        panic!("autograd failed: {}, msg: {}", x, msg)
                    }
                }
            }
        }

        let inputs = &self.node.inputs;
        let mut ret = Vec::with_capacity(inputs.len());
        for x in inputs {
            ret.push(seek(x, self.resource_store, self.feed_store, mutable));
        }
        ret
    }
}

struct NodeWithValue<'a> {
    node: &'a Tensor,
    value: OpComputeResult,
    // How many resources of this node does user requires.
    // When this is reduced to one, `value` is ready to be moved out (without copy).
    pending_count: usize,
}

impl<'a> Tensor {
    #[inline]
    fn with_value(&'a self, val: OpComputeResult) -> NodeWithValue<'a>
    {
        NodeWithValue { node: self, value: val, pending_count: 0 }
    }
}

/// Evaluates given symbolic tensors.
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
/// assert_eq!(evaluated[0], ndarray::arr1(&[0., 0.]).into_dyn());
/// assert_eq!(evaluated[1], ndarray::arr1(&[1., 1.]).into_dyn());
/// ```
pub fn eval<'a, 'b: 'a, 'c: 'a, T, U>(
    tensors: &[T],
    feeds: U,
) -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
where
    T: AsRef<Tensor>,
    U: IntoIterator<Item = &'a (&'b Tensor, &'c ndarray::Array<f32, ndarray::IxDyn>)>,
{
    // Run graph
    let feeds = feeds.into_iter().collect::<Vec<_>>();
    let mut output_storage = eval_internal(&tensors.iter().map(|t| t.as_ref()).collect(), &feeds);

    // Treat in-place or delegation ops
    let creators = tensors
        .iter()
        .map(|x| {
            let x = x.as_ref();
            let creator = if x.is_placeholder || x.persistent_array.is_some() {
                x
            } else {
                let creator = find_resource_creator(&output_storage, x);
                output_storage[creator.resource_lookup_key.get()].pending_count += 1;
                creator
            };
            creator
        })
        .collect::<Vec<&Tensor>>();

    // Shrink to fit (output_storage is moved)
    let mut key2res: BTreeMap<usize, NodeWithValue> = finalize_resource_store(output_storage);

    // Aggregate return values
    creators
        .iter()
        .map(|ref creator| {
            if let Some(ref per) = creator.persistent_array {
                // Rarely happens (case that a persistent array given by user is required)
                per.clone()
            } else if creator.is_placeholder {
                // Rarely happens (case that a feed by user is required)
                get_fed_resource(creator, &feeds).clone()
            } else {
                let res = match key2res.entry(creator.resource_lookup_key.get()) {
                    Entry::Occupied(mut ent) => {
                        // pending_count = 1, so move out
                        if ent.get().pending_count == 1 {
                            let got = ent.remove();
                            got.value.expect(got.node.op.name())
                        } else {
                            // pending_count > 1, so copy the resource
                            let got = ent.get_mut();
                            got.pending_count -= 1;
                            got.value.as_ref().expect(got.node.op.name()).clone()
                        }
                    }
                    _ => unreachable!(),
                };
                res
            }
        })
        .collect::<Vec<_>>()
}


/// Runs given symbolic tensors.
///
/// Runs, but returns nothing.
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let a = ag::variable(ndarray::arr1(&[1., 1.]));
/// let b = ag::ones(&[2]);
/// let c = ag::sub_inplace(a, &b);
///
/// // runs inplace op.
/// ag::run(&[&c], &[]);
/// // pull out shared variable
/// let should_be_zeros = &c.persistent_array;
/// assert_eq!(should_be_zeros, ndarray::arr1(&[0., 0.]).into_dyn());
///
/// ```
pub fn run<'a, 'b: 'a, 'c: 'a, T, U>(tensors: &[T], feeds: U)
where
    T: AsRef<Tensor>,
    U: IntoIterator<Item = &'a (&'b Tensor, &'c NdArray)>,
{
    // Just run the graph
    eval_internal(
        &tensors.iter().map(|t| t.as_ref()).collect(),
        &feeds.into_iter().collect(),
    );
}

#[inline]
// Recursive function which seeks a node holding the x's resource.
// Actual recursion "rarely" happens.
fn find_resource_creator<'a, 'b>(storage: &ResourceStore, x: &'b Tensor) -> &'b Tensor
{
    match storage[x.resource_lookup_key.get()].value {
        Ok(_) => x,
        Err(::OpComputeErrorStatus::Delegate { to: i }) =>
            find_resource_creator(storage, &x.inputs[i])  // recurse
        ,
        Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
            panic!("autograd failed: {}, msg: {}", x, msg),
        // TODO: Implementing
        Err(::OpComputeErrorStatus::NoOutput) =>
            unimplemented!("\"eval\" for {}", x)
    }
}


// private type alias
type OpComputeResult = Result<NdArray, ::ops::OpComputeErrorStatus>;
type ResourceStore<'a> = Vec<NodeWithValue<'a>>;
type FeedStore<'a> = Vec<&'a NdArray>;

// TODO: Use raw pointer comparison after removing "Rc"
fn get_fed_resource<'a>(node: &Tensor, feeds: &Vec<&(&Tensor, &'a NdArray)>) -> &'a NdArray
{
    // Linear search is suitable because number of feeds are so small in most cases
    for feed in feeds {
        if Rc::ptr_eq(feed.0, node) {
            return feed.1;
        }
    }
    panic!("Placeholder unfilled");
}


// Actually evaluates "endpoints".
fn eval_internal<'a>(
    endpoints: &Vec<&'a Tensor>,
    feeds: &Vec<&(&'a Tensor, &NdArray)>,
) -> ResourceStore<'a>
{
    let mut resource_store = Vec::new();
    let mut feed_store = Vec::new();

    // Obtain array resources while visiting nodes in topological order.
    // Stack-based DFS is used, to avoid stack overflow in explicit recursion.
    let mut dfs_stack: Vec<(&Tensor, bool)> = endpoints.iter().map(|&x| (x, false)).collect();
    while let Some((node, is_parent)) = dfs_stack.pop() {
        if is_parent {
            // Visit this node
            if node.is_placeholder {
                node.resource_lookup_key.set(feed_store.len());
                feed_store.push(get_fed_resource(node, &feeds));
            } else {
                node.resource_lookup_key.set(resource_store.len());
                if node.persistent_array.is_none() {
                    let y = {
                        let ctx = OpComputeContext::new(node, &resource_store, &feed_store);
                        node.with_value(node.op.compute(ctx))
                    };
                    resource_store.push(y);
                }
            }
        } else {
            // Update dfs stack
            dfs_stack.push((node, true));
            // Push children if needed
            for child in &node.inputs {
                // TODO: Use raw pointer comparison after removing "Rc"
                let visited = {
                    let k = child.resource_lookup_key.get();
                    k < resource_store.len() && Rc::ptr_eq(child, resource_store[k].node)
                };
                if !visited {
                    dfs_stack.push((child, false));
                }
            }
        }
    }
    resource_store
}


// Shrink it by dropping useless resources
// and convert it into mappings of {lookup key => resource}
fn finalize_resource_store(mut vec: ResourceStore) -> BTreeMap<usize, NodeWithValue>
{
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
    // `retained_keys` are sorted.
    // `vec` is moved into Map.
    retained_keys.into_iter().zip(vec).collect()
}


#[test]
fn test_eval()
{
    let ref v = ::ops::placeholder(&[3, 2, 1]);
    let ref z = ::ops::squeeze(v, &[2]);
    let ref g = ::ops::grad_with_default(&[z], &[v], &[&::ones(&z.shape())]);
    let eval_result = eval(g, &[(v, &::ndarray_ext::ones(&[3, 2, 1]))]);
    assert_eq!(eval_result[0].shape(), &[3, 2, 1]);
}

#[test]
fn test_constant_eval()
{
    let arr = ndarray::arr1(&[0., 0., 0.]);
    assert_eq!(arr.clone().into_dyn(), ::variable(arr).eval(&[]));
}

#[test]
fn test_placeholder_eval()
{
    let arr = ::ndarray_ext::ones(&[3, 2, 1]);
    let ref v = ::ops::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[(v, &arr)]);
    assert_eq!(eval_result[0], arr);
}

#[test]
fn test_eval_internal()
{
    let ref v = ::ops::placeholder(&[3, 2, 1]);
    let ref z = ::ops::squeeze(v, &[2]);
    let ref g = ::ops::grad_with_default(&[z], &[v], &[&::ones(&z.shape())]);
    let storage = eval_internal(&vec![&g[0]], &vec![&(v, &::ndarray_ext::ones(&[3, 2, 1]))]);

    assert_eq!(
        storage.iter().map(|x| x.node.op.name()).collect::<Vec<_>>(),
        vec![
            "ConvertToTensor",
            "Squeeze", // forward end
            "Shape",
            "StopGradient",
            "Ones",
            "ExpandDims",
        ]
    );
}
