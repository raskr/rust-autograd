/// Implements graph evaluation algorithms.
extern crate ndarray;

use context;
use ndarray_ext::NdArray;
use std::collections::LinkedList;
use std::collections::hash_map::Entry;
use std::collections::hash_map::HashMap;
use std::mem;
use tensor::Tensor;


// private type aliases
type OpComputeResult = Result<NdArray, ::OpComputeErrorStatus>;
type OutputMap = HashMap<Tensor, OpComputeResult>;
type VariableMap = HashMap<Tensor, NdArray>;



/// Evaluates input symbolic tensors.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref a = ag::zeros(&[2]);
/// let ref b = ag::ones(&[2]);
///
/// // eval two tensors at once.
/// let evaluated = ag::eval(&[a, b], &mut ag::Context::new());
/// assert_eq!(evaluated[0], ndarray::arr1(&[0., 0.]).into_dyn());
/// assert_eq!(evaluated[1], ndarray::arr1(&[1., 1.]).into_dyn());
/// ```
pub fn eval(xs: &[&Tensor], ctx: &mut context::Context)
    -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
{
    let ret = eval_tensors(xs, &mut ctx.variables, &mut ctx.outputs);
    ctx.outputs.clear();
    ret
}

/// Evaluates endpoints `tensors`.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut ctx = ag::Context::new();
/// let a = ag::variable(ndarray::arr1(&[1., 1.]), &mut ctx);
/// let b = ag::ones(&[2]);
/// let c = ag::sub_inplace(a, &b);
///
/// // runs inplace op.
/// ag::run(&[&c], &mut ctx);
/// // pull out shared variable
/// let should_be_zeros = ctx.variables.remove(&c).unwrap();
/// assert_eq!(should_be_zeros, ndarray::arr1(&[0., 0.]).into_dyn());
///
/// ```
pub fn run(tensors: &[&Tensor], ctx: &mut context::Context)
{
    eval_tensors_ref(tensors, &mut ctx.variables, &mut ctx.outputs);
}


// Recursive function which seeks array of `x` in `memo`
fn seek_array<'a>(memo: &'a OutputMap, x: &Tensor) -> &'a NdArray
{
    // safe unwrap
    match *memo.get(x).expect(&format!("Internal error: couldn't get output array of `{}`", x)) {
        Ok(ref arr) => arr,
        Err(::OpComputeErrorStatus::Delegate { to: i }) =>
            seek_array(memo, &x.inputs[i])  // hoping for x.inputs[i] to have the value
        ,
        Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
            panic!(format!("autograd failed: {}, msg: {}", x, msg))
    }
}

#[test]
fn test_list_reachable_nodes_desc()
{
    let ref v = ::zeros(&[3, 1, 2, 1]);
    let ref z = ::squeeze(v, &[3, 1]);
    let ref g = ::grad_with_default(&[z], &[v], &[&::ones(&z.shape())])[0];
    assert_eq!(
        list_reachable_nodes_desc(g)
            .iter()
            .rev()
            .map(|a| a.op.name())
            .collect::<Vec<_>>(),
        vec![
            "ConvertToTensor",
            "Zeros",
            "ConvertToTensor",
            "Squeeze",
            "Shape",
            "StopGradients",
            "Ones",
            "ConvertToTensor",
            "ExpandDims",
        ]
    );
}

// - Lists all reachable nodes from a "sink" node
// - Sorted in "descending" order
// - Variables/Constants are omitted
fn list_reachable_nodes_desc<'a>(target: &'a Tensor) -> LinkedList<&'a Tensor>
{
    let mut stack: Vec<&Tensor> = vec![target];
    let mut result: LinkedList<&Tensor> = LinkedList::new();
    // DFS (Avoid recursion)
    while let Some(pop) = stack.pop() {
        let name = pop.op.name();
        if name == "Variable" || name == "Const" {
            continue;
        }
        result.push_back(pop);
        for x in pop.inputs.iter() {
            stack.push(x);
        }
    }
    result
}

#[doc(hidden)]
// Performs actual graph traversal and its evaluation.
// Evaluated output arrays are cached in `memo`.
fn perform_eval(target: &Tensor, vars: &mut VariableMap, memo: &mut OutputMap)
{
    // Iterate all nodes that are reachable from a sink node, in ascending topological order.
    for cur in list_reachable_nodes_desc(target).into_iter().rev() {
        if vars.contains_key(cur) || memo.contains_key(cur) {
            continue;
        }
        let y: Option<OpComputeResult> = {
            // ** make xs **
            let inputs = &cur.inputs;
            let xs = inputs
                .iter()
                .map(|x| if let Some(a) = vars.get(x) {
                    a
                } else {
                    seek_array(memo, x)
                })
                .collect::<Vec<_>>();

            // ** compute output **
            if cur.op.inplace() {
                // make xs mutable temporarily
                let mut xs: Vec<&mut NdArray> = unsafe { mem::transmute(xs) };
                if let Err(::OpComputeErrorStatus::BadInput(msg)) =
                    cur.op.compute_inplace(xs.as_mut_slice())
                {
                    // For inplace ops, reports error here
                    panic!(msg)
                }
                None
            } else {
                // non-inplace op
                println!("{}", vars.len());
                // ここで、Variable.compute() が呼ばれる
                Some(cur.op.compute(xs.as_slice()))
            }
        };

        // ** cache the output **
        if let Some(y_) = y {
            // normal op
            memo.insert(cur.clone(), y_);
        } else {
            // inplace op
            let x = &cur.inputs[0]; // get x as a output
            if let Some(y) = memo.remove(x) {
                // move array from memo
                memo.insert(cur.clone(), y);
            } else {
                // move array from variable set
                if let Some(y) = vars.remove(x) {
                    vars.insert(cur.clone(), y);
                } else {
                    unreachable!()
                }
            }
        }

    }
}

// Recursive function which seeks the owner node of `x` in `memo`
fn seek_array_owner<'a, 'b>(memo: &'a OutputMap, x: &'b Tensor) -> &'b Tensor
{
    if let Some(x_) = memo.get(x) {
        match *x_ {
            Ok(_) => x,
            Err(::OpComputeErrorStatus::Delegate { to: i }) =>
                seek_array_owner(memo, &x.inputs[i])  // hoping for x.inputs[i] to have the value
            ,
            Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
                panic!(format!("autograd failed: {}, msg: {}", x, msg))
        }
    } else {
        // `x` is owner but array is already took out by past self; so returns
        // self again.
        x
    }
}


#[doc(hidden)]
// TODO: clean code
// limited to internal use.
pub fn eval_tensors(
    tensors: &[&Tensor],
    variables: &mut VariableMap,
    memo: &mut OutputMap,
) -> Vec<NdArray>
{
    // run graph
    for &t in tensors.iter() {
        perform_eval(t, variables, memo);
    }

    // `usize` is number of owners of the array
    let mut owner2arr = HashMap::<&Tensor, (NdArray, usize)>::new();
    let mut owners = Vec::with_capacity(tensors.len());

    // build owner2arr and owners
    for &t in tensors.iter() {
        if let Some(var) = variables.get(t) {
            // case of "from variable set"
            match owner2arr.entry(t) {
                Entry::Occupied(mut ent) => {
                    // increment shared count
                    ent.get_mut().1 += 1
                }
                Entry::Vacant(ent) => {
                    ent.insert((var.clone(), 1));
                }
            }
            owners.push(t);
        } else {
            // case of "from output memo"
            let owner = seek_array_owner(memo, t);
            match owner2arr.entry(owner) {
                Entry::Occupied(mut ent) => {
                    // increment shared count
                    ent.get_mut().1 += 1
                }
                Entry::Vacant(ent) => {
                    ent.insert((memo.remove(owner).unwrap().unwrap(), 1));
                }
            }
            owners.push(owner);
        };
    }

    // return arrays
    owners
        .into_iter()
        .map(move |owner| {
            if let Some(arr) = owner2arr.get_mut(owner).and_then(|&mut (ref arr,
                   ref mut shared_count)| {
                if *shared_count >= 2 {
                    *shared_count -= 1;
                    Some(arr)
                } else {
                    None
                }
            })
            {
                // Shared count is over 2, so
                // clone the array and exit this closure.
                return arr.clone();
            }
            // Otherwise, shared count is 1, then remove the array from map and return it.
            owner2arr.remove(owner).unwrap().0
        })
        .collect::<Vec<NdArray>>()
}


/// Evaluates endpoints `tensors` and returns the "references" to their arrays.
pub fn eval_tensors_ref<'a>(
    tensors: &[&Tensor],
    variables: &'a mut VariableMap,
    memo: &'a mut OutputMap,
) -> Vec<&'a NdArray>
{
    // run graph
    for t in tensors.iter() {
        perform_eval(t, variables, memo);
    }

    let mut results = Vec::with_capacity(tensors.len());
    for t in tensors.iter() {
        if let Some(var) = variables.get(t) {
            results.push(var);
        } else {
            // case of "from output memo"
            let owner = seek_array_owner(memo, t);
            results.push(memo.get(owner).unwrap().as_ref().unwrap());
        };
    }
    results
}
