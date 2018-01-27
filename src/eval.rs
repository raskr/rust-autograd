// Implements graph evaluation algorithms.
extern crate ndarray;

use ndarray_ext::NdArray;
use std::collections::BTreeMap;
use std::collections::btree_map::Entry;
use std::mem;
use std::rc::Rc;
use tensor::Tensor;


// Module private.
struct TensorWithResource<'a> {
    inner: &'a Tensor,
    // Evaluation result of this tensor.
    val: OpComputeResult,
    // How many resources of this tensor does user requires.
    // If this is reduced to one, `val` can be moved.
    pending_count: usize,
}

impl<'a> Tensor {
    #[inline]
    fn with_resource(&'a self, val: OpComputeResult) -> TensorWithResource<'a>
    {
        TensorWithResource { inner: self, val, pending_count: 0 }
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
    let mut key2res: BTreeMap<usize, TensorWithResource> = finalize_output_storage(output_storage);

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
                            got.val.expect(got.inner.op.name())
                        } else {
                            // pending_count > 1, so copy the resource
                            let got = ent.get_mut();
                            got.pending_count -= 1;
                            got.val.as_ref().expect(got.inner.op.name()).clone()
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
// Recursive function which seeks array resource of `x` in 'output_storage` or `feed_storage`.
// Actual recursion "rarely" happens.
fn find_resource<'a, 'b: 'a>(
    output_storage: &'a OutputStorage,
    feed_storage: &FeedStorage<'a>,
    x: &'b Tensor,
    inplace: bool,
) -> &'a NdArray
{
    if let Some(ref per) = x.persistent_array {
        assert!(!inplace || x.op.name() != "Const");
        per
    } else if x.is_placeholder {
        feed_storage[x.resource_lookup_key.get()]
    } else if x.op.inplace() {
        find_resource(output_storage, feed_storage, &x.inputs[0], inplace)
    } else {
        match output_storage[x.resource_lookup_key.get()].val {
            Ok(ref arr) => arr,
            // hoping for x.inputs[i] to have the value
            Err(::OpComputeErrorStatus::Delegate { to: i }) => {
                find_resource(output_storage, feed_storage, &x.inputs[i], inplace)
            }
            // panic
            Err(::OpComputeErrorStatus::BadInput(ref msg)) => {
                panic!("autograd failed: {}, msg: {}", x, msg)
            }
        }
    }
}


#[inline]
// Recursive function which seeks a node holding the x's resource.
// Actual recursion "rarely" happens.
fn find_resource_creator<'a, 'b>(storage: &OutputStorage, x: &'b Tensor) -> &'b Tensor
{
    if x.op.inplace() {
        find_resource_creator(storage, &x.inputs[0])
    } else {
        match storage[x.resource_lookup_key.get()].val {
            Ok(_) => x,
            Err(::OpComputeErrorStatus::Delegate { to: i }) =>
                find_resource_creator(storage, &x.inputs[i])  // recurse
            ,
            Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
                panic!("autograd failed: {}, msg: {}", x, msg),
        }
    }
}


// private type alias
type OpComputeResult = Result<NdArray, ::ops::OpComputeErrorStatus>;
type OutputStorage<'a> = Vec<TensorWithResource<'a>>;
type FeedStorage<'a> = Vec<&'a NdArray>;

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
) -> OutputStorage<'a>
{
    // Use vector for resource storage to do O(1) lookup.
    // Storage sizes are unknown at this point.
    let mut output_storage: OutputStorage<'a> = Vec::new();
    let mut feed_storage: Vec<&NdArray> = Vec::new();

    // Obtain array resources while visiting nodes in topological order.
    // Stack-based DFS is used, to avoid stack overflow in explicit recursion.
    let mut dfs_stack: Vec<(&Tensor, bool)> = endpoints.iter().map(|&x| (x, false)).collect();
    while let Some((node, is_parent)) = dfs_stack.pop() {
        if is_parent {
            // Visit this node
            if node.is_placeholder {
                node.resource_lookup_key.set(feed_storage.len());
                feed_storage.push(get_fed_resource(node, &feeds));
            } else {
                node.resource_lookup_key.set(output_storage.len());
                if node.persistent_array.is_none() {
                    let y = compute_y(node, &output_storage, &feed_storage);
                    preserve_y(node, y, &mut output_storage);
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
                    k < output_storage.len() && Rc::ptr_eq(child, output_storage[k].inner)
                };
                if !visited {
                    dfs_stack.push((child, false));
                }
            }
        }
    }
    output_storage
}


#[inline]
fn compute_y(
    node: &Tensor,
    output_storage: &OutputStorage,
    feed_storage: &FeedStorage,
) -> Option<OpComputeResult>
{
    let is_inplace = node.op.inplace();
    // make xs
    let xs = node.inputs
        .iter()
        .map(|x| find_resource(output_storage, feed_storage, x, is_inplace))
        .collect::<Vec<_>>();

    // compute output
    if !is_inplace {
        Some(node.op.compute(xs.as_slice()))
    } else {
        // make xs mutable temporarily
        let mut xs: Vec<&mut NdArray> = unsafe { mem::transmute(xs) };
        if let Err(::OpComputeErrorStatus::BadInput(msg)) =
            node.op.compute_inplace(xs.as_mut_slice())
        {
            panic!(msg)
        }
        None
    }
}


#[inline]
fn preserve_y<'a>(
    node: &'a Tensor,
    node_y: Option<OpComputeResult>,
    output_storage: &mut OutputStorage<'a>,
)
{
    if let Some(y) = node_y {
        // push back
        output_storage.push(node.with_resource(y));
    } else {
        // inplace op: just transfer the lookup key (node.inputs[0] => node)
        node.resource_lookup_key.set(
            node.inputs[0]
                .resource_lookup_key
                .get(),
        );
    }
}


// Shrink output storage by dropping useless resources
// and convert it into mappings of {lookup key => resource}
fn finalize_output_storage(mut vec: OutputStorage) -> BTreeMap<usize, TensorWithResource>
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
        storage
            .iter()
            .map(|x| x.inner.op.name())
            .collect::<Vec<_>>(),
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
