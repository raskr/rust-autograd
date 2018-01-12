// Implements graph evaluation algorithms.
extern crate ndarray;

use ndarray_ext::NdArray;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::mem;
use tensor::Tensor;


#[doc(hidden)]
// Contains properties used for evaluation of a tensor.
pub struct TensorEvaluationContext {
    // Key of a tensor to look up corresponding array resource in a storage.
    pub resource_lookup_key: Cell<usize>,
    // Immutable flag of tensor is placeholder or not.
    pub is_placeholder: bool,
    // How many resources of a tensor does user requires.
    pub num_required_as_ret_val: Cell<usize>,
}

impl TensorEvaluationContext {
    pub fn new(is_placeholder: bool) -> TensorEvaluationContext
    {
        TensorEvaluationContext {
            resource_lookup_key: Cell::new(!0),
            is_placeholder,
            num_required_as_ret_val: Cell::new(0),
        }
    }

    #[inline]
    // Helper
    pub fn decrement_shared_count(&self)
    {
        self.num_required_as_ret_val.set(
            self.num_required_as_ret_val
                .get() - 1,
        )
    }

    #[inline]
    // Helper
    pub fn increment_shared_count(&self)
    {
        self.num_required_as_ret_val.set(
            self.num_required_as_ret_val
                .get() + 1,
        )
    }
}


#[inline]
fn remove_or_borrow_resource(
    node: &Tensor,
    key: &usize,
    map: &mut BTreeMap<usize, OutputStorageContent>,
    remove: bool,
) -> NdArray
{
    // Safe unwrapping is guarantied by topological ordering.
    if remove {
        match map.remove(key).unwrap().1 {
            Ok(arr) => arr,
            Err(::OpComputeErrorStatus::BadInput(ref msg)) => panic!(bad_input_msg(node, msg)),
            _ => unreachable!(),
        }
    } else {
        match map.get(key).unwrap().1 {
            Ok(ref arr) => arr.clone(),
            Err(::OpComputeErrorStatus::BadInput(ref msg)) => panic!(bad_input_msg(node, msg)),
            _ => unreachable!(),
        }
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
    let output_storage = eval_internal(&tensors.iter().map(|t| t.as_ref()).collect(), &feeds);

    // Treat in-place or delegation ops
    let creators = tensors
        .iter()
        .map(|x| {
            let creator = find_resource_creator(&output_storage, x.as_ref());
            creator.eval_context.increment_shared_count();
            creator
        })
        .collect::<Vec<&Tensor>>();

    // Shrink to fit
    let mut key2res: BTreeMap<usize, OutputStorageContent> =
        finalize_output_storage(output_storage);

    // Aggregate return values
    creators
        .iter()
        .map(|ref creator| {
            let ctx = &creator.eval_context;
            if let Some(ref per) = creator.persistent_array {
                // Rarely happens (case that user want a constant value)
                per.clone()
            } else if ctx.is_placeholder {
                // Rarely happens (case that user want a placeholder value)
                get_fed_resource(creator, &feeds).clone()
            } else {
                let res = remove_or_borrow_resource(
                    creator,
                    &ctx.resource_lookup_key.get(),
                    &mut key2res,
                    ctx.num_required_as_ret_val.get() == 1,
                );
                ctx.decrement_shared_count();
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

#[inline(always)]
fn key_of(x: &Tensor) -> usize
{
    x.eval_context.resource_lookup_key.get()
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
    } else if x.eval_context.is_placeholder {
        feed_storage[key_of(x)]
    } else if x.op.inplace() {
        find_resource(output_storage, feed_storage, &x.inputs[0], inplace)
    } else {
        match output_storage[key_of(x)].1 {
            Ok(ref arr) => arr,
            // hoping for x.inputs[i] to have the value
            Err(::OpComputeErrorStatus::Delegate { to: i }) => {
                find_resource(output_storage, feed_storage, &x.inputs[i], inplace)
            }
            // panic
            Err(::OpComputeErrorStatus::BadInput(ref msg)) => panic!(bad_input_msg(x, msg)),
        }
    }
}


fn bad_input_msg(x: &Tensor, msg: &str) -> String
{
    format!("autograd failed: {}, msg: {}", x, msg)
}


#[inline]
// Recursive function which seeks a node holding the x's resource.
// Actual recursion "rarely" happens.
fn find_resource_creator<'a, 'b>(storage: &OutputStorage, x: &'b Tensor) -> &'b Tensor
{
    if x.eval_context.is_placeholder || x.persistent_array.is_some() {
        x
    } else {
        if x.op.inplace() {
            find_resource_creator(storage, &x.inputs[0])
        } else {
            match storage[key_of(x)].1 {
                Ok(_) => x,
                Err(::OpComputeErrorStatus::Delegate { to: i }) =>
                    find_resource_creator(storage, &x.inputs[i])  // recurse
                ,
                Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
                    panic!(bad_input_msg(x, msg))
            }
        }
    }
}


// private type alias
type OpComputeResult = Result<NdArray, ::OpComputeErrorStatus>;
type OutputStorageContent<'a> = (&'a Tensor, OpComputeResult);
type OutputStorage<'a> = Vec<OutputStorageContent<'a>>;
type FeedStorage<'a> = Vec<&'a NdArray>;

use std::rc::Rc;

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


// Actually evaluates tensors "from".
fn eval_internal<'a>(
    from: &Vec<&'a Tensor>,
    feeds: &Vec<&(&'a Tensor, &NdArray)>,
) -> OutputStorage<'a>
{
    // Use vector for resource storage to do O(1) lookup.
    // Storage sizes are unknown at this point.
    let mut output_storage: Vec<OutputStorageContent> = Vec::new();
    let mut feed_storage: Vec<&NdArray> = Vec::new();

    // Obtain array resources while visiting nodes in execution (topological) order.
    // Stack-based DFS is used, to avoid stack overflow in explicit recursion.
    let mut dfs_stack: Vec<(&Tensor, bool)> = from.iter().map(|&x| (x, false)).collect();
    while let Some((node, is_parent)) = dfs_stack.pop() {
        let ctx: &TensorEvaluationContext = &node.eval_context;
        if is_parent {
            // Visit this node
            if ctx.is_placeholder {
                let res = get_fed_resource(node, &feeds);
                ctx.resource_lookup_key.set(feed_storage.len());
                feed_storage.push(res);
            } else {
                ctx.resource_lookup_key.set(output_storage.len());
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
                let k = key_of(child);
                // TODO: Use raw pointer comparison after removing "Rc"
                let visited = k < output_storage.len() && Rc::ptr_eq(child, output_storage[k].0);
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
        .map(|x| {
            find_resource(output_storage, feed_storage, x, is_inplace)
        })
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
        output_storage.push((node, y));
    } else {
        // inplace op: just transfer the lookup key (node.inputs[0] => node)
        node.eval_context.resource_lookup_key.set(
            node.inputs[0]
                .eval_context
                .resource_lookup_key
                .get(),
        );
    }
}


// Shrink output storage by dropping useless resources
// and converts it into mapping of {lookup key => resource}
fn finalize_output_storage(mut vec: OutputStorage) -> BTreeMap<usize, OutputStorageContent>
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
                let node: &Tensor = v[i].0;
                let ctx = &node.eval_context;

                if ctx.num_required_as_ret_val.get() == 0 {
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
    let ref v = ::placeholder(&[3, 2, 1]);
    let ref z = ::squeeze(v, &[2]);
    let ref g = ::grad_with_default(&[z], &[v], &[&::ones(&z.shape())]);
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
    let ref v = ::placeholder(&[3, 2, 1]);
    let eval_result = eval(&[v], &[(v, &arr)]);
    assert_eq!(eval_result[0], arr);
}

#[test]
fn test_eval_internal()
{
    let ref v = ::placeholder(&[3, 2, 1]);
    let ref z = ::squeeze(v, &[2]);
    let ref g = ::grad_with_default(&[z], &[v], &[&::ones(&z.shape())]);
    let storage = eval_internal(&vec![&g[0]], &vec![&(v, &::ndarray_ext::ones(&[3, 2, 1]))]);

    assert_eq!(
        storage.iter().map(|x| x.0.op.name()).collect::<Vec<_>>(),
        vec![
            "ConvertToTensor",
            "Squeeze", // forward end
            "Shape",
            "StopGradients",
            "Ones",
            "ExpandDims",
        ]
    );
}
