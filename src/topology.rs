extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops;
use std::collections::binary_heap::BinaryHeap;
use std::collections::hash_set::HashSet;
use std::mem;
use tensor::Tensor;



/// Performs actual graph traversal and its evaluation
// TODO: loop-based rather than recursion (this would be difficult)
#[inline]
pub fn perform_eval(target: &Tensor, memo: &mut FnvHashMap<Tensor, NdArray>, train: bool)
{
    if memo.contains_key(target) {
        return;
    }

    let y = {
        let mut xs = vec![];
        {
            let ref inputs = target.borrow().inputs;
            // integrating loops below is impossible because of
            // "memo is already mutably borrowed"
            for input in inputs.iter() {
                perform_eval(input, memo, train);
            }
            for input in inputs.iter() {
                // unwrap is safe
                xs.push(memo.get(input).unwrap());
            }
        }
        // run op
        target.borrow_mut().op.compute(xs.as_slice(), train)
    };

    // cache output
    memo.insert(target.clone(), y);
}


#[inline]
// make mapping of {node => the node contributed to gradient or not}
pub fn contributed_to_grads(objective: &Tensor, variables: &[&Tensor]) -> FnvHashMap<Tensor, bool>
{
    fn rec(target: &Tensor, vars: &[&Tensor], memo: &mut FnvHashMap<Tensor, bool>)
    {
        if memo.contains_key(target) {
            return;
        }

        let mut contrib = false;

        if vars.contains(&target) {
            contrib = true;
        } else {
            for input in target.borrow().inputs.iter() {
                // recurse
                rec(input, vars, memo);
                // unwrap is always safe
                contrib |= *memo.get(input).unwrap();
            }
        }

        memo.insert(target.clone(), contrib);
    }

    let mut memo = FnvHashMap::default();
    rec(objective, variables, &mut memo);
    memo
}


#[inline]
/// Returns symbolic gradient tensors.
///
/// This computes partial derivatives of `objective` with `variables` and returns the
/// gradients. This is achieved by building the subgraph between `objective` and
/// `variables` in reverse order from user's graph definition.
///
/// NOTE: Nodes that do not contribute to the gradient won't be included to avoid
/// unnecessary computation.
pub fn symbolic_gradients(
    objective: &Tensor,
    variables: &[&Tensor],
    initial_grad: Option<&Tensor>,
) -> Vec<Tensor>
{
    #[inline]
    fn maybe_reduce_grad(gys: &mut Vec<Tensor>)
    {
        if gys.len() < 2 {
            return;
        }
        let acc = {
            // values to refs
            let refs = gys.iter().map(|a| a).collect::<Vec<_>>();
            if refs.len() == 2 {
                // normal addition
                refs[0] + refs[1]
            } else {
                // For 3 or more gradients, AddN, i.e. inplace accumulation
                // is preferred for performance
                ops::add_n(refs.as_slice())
            }
        };
        mem::swap(&mut vec![acc], gys);
    }

    let initial_grad = initial_grad.map(|a| a.clone()).unwrap_or_else(
        || ops::scalar(1.),
    );

    // Mapping of {y => [gy]}
    let mut grads = FnvHashMap::default();

    // Mapping of {node => must visit or not (boolean)}
    let contrib = contributed_to_grads(objective, variables);

    // Prepare a heap with tensor's rank numbers for reversed
    // topological sort.
    let mut heap = BinaryHeap::new();
    heap.push(objective.clone());
    grads.insert(objective.clone(), vec![initial_grad]);

    // This prevents calling `grad()` twice or more
    let mut grad_done = HashSet::<Tensor>::new();

    // builds backward graph
    while let Some(target) = heap.pop() {
        let borrowed_target = target.borrow();

        // Vec<Tensor> to Vec<&Tensor>
        let xs = borrowed_target.inputs.iter().map(|a| a).collect::<Vec<_>>();

        // time to call `grad`
        let gxs = {
            if let Some(gys) = grads.get_mut(&target) {
                maybe_reduce_grad(gys);
                borrowed_target.op.grad(&gys[0], xs.as_slice(), &target)
            } else {
                unreachable!("Safe unwrapping is guaranteed by topological ordering")
            }
        };

        debug_assert_eq!(
            xs.len(),
            gxs.len(),
            "Wrong `grad` implementation of {}.",
            target
        );

        for (x, maybe_gx) in xs.into_iter().zip(gxs) {
            if !contrib.contains_key(x) {
                continue;
            }
            // cuts the backward path if gx is None.
            if let Some(gx) = maybe_gx {
                // memo gx
                if let Some(mut gys) = grads.remove(x) {
                    // gradient accumulation should be delayed here
                    gys.push(gx);
                    grads.insert(x.clone(), gys);
                } else {
                    grads.insert(x.clone(), vec![gx]);
                }
                // update heap
                if !x.is_source() && !grad_done.contains(x) {
                    grad_done.insert(x.clone());
                    heap.push(x.clone());
                }
            }
        }
    }

    variables
        .iter()
        .map(|v| {
            let mut gys = grads.remove(v).expect(
                "Input variable(s) didn't contributed to gradient computation",
            );
            maybe_reduce_grad(&mut gys);
            gys.remove(0)
        })
        .collect::<Vec<Tensor>>()
}
