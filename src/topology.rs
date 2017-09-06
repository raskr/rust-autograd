extern crate ndarray;

use std::mem;
use std::collections::hash_map::HashMap;
use std::collections::hash_set::HashSet;
use std::collections::binary_heap::BinaryHeap;
use tensor::{Tensor, Input};
use ndarray_ext::NdArray;
use ops;


/// Performs actual graph traversal and its evaluation
// TODO: loop-based rather than recursion
#[inline]
pub fn perform_eval(target: &Tensor, memo: &mut HashMap<Tensor, NdArray>, train: bool) {
    if memo.contains_key(target) {
        return;
    }

    // need clone
    let ref inputs = target.borrow().inputs.clone();

    // recurse
    for x in inputs.iter() {
        perform_eval(x, memo, train);
    }

    let y = {
        let mut xs = Vec::with_capacity(inputs.len());
        for input in inputs.iter() {
            xs.push(memo.get(input).unwrap());
        }
        // run op
        target.borrow_mut().op.compute(xs.as_slice(), train)
    };

    // cache output
    memo.insert(target.clone(), y);
}


#[inline]
// make mapping of {node => the node contributed to gradient or not}
// TODO: loop-based rather than recursion
pub fn contributed_to_grads(objective: &Tensor, wrt: &[&Tensor]) -> HashMap<Tensor, bool> {
    fn rec(target: &Tensor, wrt: &[&Tensor], memo: &mut HashMap<Tensor, bool>) {
        if memo.contains_key(target) {
            return;
        }

        let mut contributed = false;

        if wrt.contains(&target) {
            contributed = true;
        } else {
            for input in target.borrow().inputs.iter() {
                // recurse
                rec(input, wrt, memo);
                // unwrap is always safe
                contributed |= *memo.get(input).unwrap();
            }
        }

        memo.insert(target.clone(), contributed);
    }

    let mut memo = HashMap::new();
    rec(objective, wrt, &mut memo);
    memo
}


#[inline]
/// Returns symbolic gradient tensors.
///
/// This computes partial derivatives of `objective` with `wrt` and returns the
/// gradients. This is achieved by building the subgraph between `objective` and
/// `wrt` in reverse order from user's graph definition.
///
/// NOTE: Nodes that do not contribute to the gradient won't be included to avoid
/// unnecessary computation.
pub fn symbolic_gradients(
    objective: &Tensor,
    wrt: &[&Tensor],
    initial_grad: Option<&Tensor>,
) -> Vec<Tensor> {
    let initial_grad = initial_grad.map(|a| a.clone()).unwrap_or_else(||
        ::graph_sources::scalar(1.)
    );

    // Mapping of {y => gy}
    let mut grads = HashMap::new();

    // Mapping of {node => must visit or not (boolean)}
    let contrib = contributed_to_grads(objective, wrt);

    // Prepare the heap with tensor's rank numbers for reversed
    // topological sort.
    let mut heap = BinaryHeap::new();
    heap.push(objective.clone());
    grads.insert(objective.clone(), initial_grad);
    let mut lop_done = HashSet::<Tensor>::new();

    // backprop implementation
    while let Some(target) = heap.pop() {
        let target_brr = target.borrow();

        let mut xs = vec![];
        for x in target_brr.inputs.iter() {
            xs.push(x);
        }

        let gxs = {
            // TODO: remove unwrap
            // Calling the Lop of `op`, whose arguments are:
            // "op's inputs", "op's output", "the output's gradient"
            let gy = grads.get(&target).unwrap();
            target_brr.op.lop(gy, xs.as_slice(), &target)
        };

        debug_assert_eq!(xs.len(), gxs.len(), "Bad grad from ({})", target);

        // For now, if gx is None, cutting the backward path
        for (x, maybe_gx) in xs.iter().zip(gxs.iter()) {
            if let Some(gx) = maybe_gx.as_ref() {
                if contrib.contains_key(x) {
                    // memo gradient
                    if let Some(g_base) = grads.remove(x) {
                        // accumulate
                        grads.insert((*x).clone(), g_base + gx);
                    } else {
                        // first time
                        grads.insert((*x).clone(), (*gx).clone());
                    }
                    // update heap
                    if !x.is_source() && !lop_done.contains(x).clone() {
                        lop_done.insert((*x).clone());
                        heap.push((*x).clone());
                    }
                }
            }
        }
    }

    // TODO: returning appropriate error
    wrt.iter()
       .map(|v|
           grads.remove(v).expect("Input variable(s) didn't contributed to gradient computation")
       )
       .collect::<Vec<Tensor>>()
}


pub fn collect_nodes_from(end_point: &Tensor) -> HashSet<Tensor> {
    let mut collected = HashSet::new();
    end_point.visit_once(&mut |arg| {
        collected.insert(arg.clone());
    });
    collected
}
