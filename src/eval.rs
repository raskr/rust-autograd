/// Implements graph evaluation algorithms.
extern crate ndarray;

use context;
use ndarray_ext::NdArray;
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
/// let ref a = ag::zeros(&[2, 2]);
/// let ref b = ag::ones(&[2, 2]);
///
/// // eval two tensors at once.
/// let evaluated = ag::eval(&[a, b], &mut ag::Context::new());
/// assert_eq!(evaluated[0], ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn());
/// assert_eq!(evaluated[1], ndarray::arr2(&[[1., 1.], [1., 1.]]).into_dyn());
/// ```
pub fn eval(xs: &[&Tensor], ctx: &mut context::Context)
    -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
{
    let xs = xs.into_iter().map(|&a| a.clone()).collect::<Vec<_>>();
    let ret = eval_tensors(xs.as_slice(), &mut ctx.variables, &mut ctx.outputs);
    ctx.outputs.clear();
    ret
}


// Recursive function which seeks array of `x` in `memo`
fn seek_array<'a>(memo: &'a OutputMap, x: &Tensor) -> &'a NdArray
{
    // safe unwrap
    match *memo.get(x).unwrap() {
        Ok(ref arr) => arr,
        Err(::OpComputeErrorStatus::Delegate { to: i }) =>
            seek_array(memo, &x.inputs[i])  // hoping for x.inputs[i] to have the value
        ,
        Err(::OpComputeErrorStatus::BadInput(ref msg)) =>
            panic!(format!("autograd failed: {}, msg: {}", x, msg))
    }
}


#[doc(hidden)]
// Performs actual graph traversal and its evaluation.
// Evaluated output arrays are cached in `memo`.
// TODO: loop-based rather than recursion
pub fn perform_eval(target: &Tensor, vars: &mut VariableMap, memo: &mut OutputMap)
{

    if vars.contains_key(target) || memo.contains_key(target) {
        return;
    }

    let inputs = &target.inputs;

    for x in inputs.iter() {
        perform_eval(x, vars, memo);
    }

    let y: Option<OpComputeResult> = {
        // ** make xs **
        let mut xs = Vec::with_capacity(inputs.len());
        for x in inputs.iter() {
            if let Some(a) = vars.get(x) {
                // from variable set
                xs.push(a);
            } else {
                // from memo set
                xs.push(seek_array(memo, x));
            }
        }

        // ** compute output **
        if target.op.inplace() {
            // make xs mutable temporarily
            let mut xs: Vec<&mut NdArray> = unsafe { mem::transmute(xs) };
            if let Err(::OpComputeErrorStatus::BadInput(msg)) =
                target.op.compute_inplace(xs.as_mut_slice())
            {
                // For inplace ops, reports error here
                panic!(msg)
            }
            None
        } else {
            // non-inplace op
            Some(target.op.compute(xs.as_slice()))
        }
    };

    // ** cache the output **
    if let Some(y_) = y {
        // normal op
        memo.insert(target.clone(), y_);
    } else {
        let x = &target.inputs[0]; // inplace => get x as a output
        if let Some(y) = memo.remove(x) {
            // move array from memo
            memo.insert(target.clone(), y);
        } else {
            // move array from variable set
            if let Some(y) = vars.remove(x) {
                vars.insert(target.clone(), y);
            } else {
                unreachable!()
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
        // `x` is owner but array is already took out by past self.
        x
    }
}


// TODO: clean code
#[doc(hidden)]
// limited to internal use.
pub fn eval_tensors(
    tensors: &[Tensor],
    variables: &mut VariableMap,
    memo: &mut OutputMap,
) -> Vec<NdArray>
{
    // run graph
    for t in tensors.iter() {
        perform_eval(t, variables, memo);
    }

    // `usize` is number of owners of the array
    let mut owner2arr = HashMap::<&Tensor, (NdArray, usize)>::default();
    let mut owners = Vec::with_capacity(tensors.len());

    // build owner2arr and owners
    for t in tensors.iter() {
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
            if let Some(arr) = owner2arr.get_mut(owner).and_then(
                |&mut (ref arr, ref mut count)| {
                    if *count >= 2 {
                        *count -= 1;
                        Some(arr)
                    } else {
                        None
                    }
                },
            )
            {
                // clone the array and exit this closure
                return arr.clone();
            }
            // otherwise, shared count is 1, then remove the array from map and return it
            owner2arr.remove(owner).unwrap().0
        })
        .collect::<Vec<NdArray>>()
}
