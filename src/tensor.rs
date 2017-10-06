extern crate ndarray;
extern crate fnv;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops;
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::hash_set::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops::{Deref, DerefMut};
use std::rc::Rc;


/// Symbolic multi-dimensional array which supports
/// efficient gradient computation.
pub struct Tensor(pub Rc<RefCell<RawTensor>>);

pub struct RawTensor {
    /// Operation of this node
    pub op: Box<ops::Op>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// rank number for topological ordering
    pub rank: usize,
}


impl Tensor {
    #[inline]
    /// Returns true if this node has no incoming nodes.
    pub fn is_source(&self) -> bool
    {
        self.borrow().inputs.is_empty()
    }

    #[inline]
    pub fn visit_once<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        self.run_visit_once(f, &mut HashSet::new())
    }

    #[inline]
    fn run_visit_once<F>(&self, f: &mut F, visited: &mut HashSet<Tensor>)
    where
        F: FnMut(&Tensor) -> (),
    {
        if visited.contains(self) {
            return; // exit early
        } else {
            visited.insert(self.clone()); // first visit
        }

        f(self);

        for child in &(*self).borrow().inputs {
            child.run_visit_once(f, visited)
        }
    }

    #[inline]
    pub fn visit<F>(&self, f: &mut F)
    where
        F: FnMut(&Tensor) -> (),
    {
        f(self);

        for child in &(*self).borrow().inputs {
            child.visit(f)
        }
    }
}

impl Ord for Tensor {
    /// Compares the ranks in topological ordering
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.borrow().rank.cmp(&other.borrow().rank)
    }
}

impl PartialOrd for Tensor {
    /// Compares the ranks in topological ordering
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}

// empty implementation
impl Eq for Tensor {}

impl PartialEq for Tensor {
    fn eq(&self, other: &Tensor) -> bool
    {
        // compare addresses on the heap
        Rc::ptr_eq(&self.0, &other.0)
    }
}

// empty implementation
impl Hash for Tensor {
    fn hash<H: Hasher>(&self, _: &mut H)
    {
    }
}

// data is not cloned; only reference count is incremented.
impl Clone for Tensor {
    fn clone(&self) -> Tensor
    {
        Tensor(self.0.clone())
    }
}

impl Deref for Tensor {
    type Target = Rc<RefCell<RawTensor>>;
    fn deref(&self) -> &Self::Target
    {
        &self.0
    }
}

impl DerefMut for Tensor {
    fn deref_mut<'a>(&'a mut self) -> &'a mut Rc<RefCell<RawTensor>>
    {
        &mut self.0
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
    {
        let ref_obj = &self.0.borrow();
        let input_names = ref_obj
            .inputs
            .iter()
            .map(|a| a.borrow().op.name().to_string())
            .collect::<Vec<String>>();
        write!(
            f,
            "op: {}\ninputs: {:?}\n",
            ref_obj.op.name(),
            input_names.as_slice()
        )
    }
}


#[doc(hidden)]
#[inline]
pub fn eval_tensors(
    tensors: &[Tensor],
    variables: &FnvHashMap<Tensor, NdArray>,
    memo: &mut FnvHashMap<Tensor, NdArray>,
) -> Vec<NdArray>
{
    // run graph
    for t in tensors.iter() {
        ::topology::perform_eval(t, variables, memo, true);
    }

    // extracts target arrays
    let mut evaluated_arrays = Vec::with_capacity(tensors.len());

    for (i, t) in tensors.iter().enumerate() {
        // Need to handle cases where multiple gradient nodes
        // share an output array, and `t` is a variable.
        // (Safe unwrapping is guaranteed by ::topology::symbolic_gradients)
        let contains = tensors[i + 1..].contains(t);
        let in_memo = memo.contains_key(t);
        match (contains, in_memo) {
            (true, true) => evaluated_arrays.push(memo.get(t).unwrap().clone()),
            (true, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
            (false, true) => evaluated_arrays.push(memo.remove(t).unwrap()),
            (false, false) => evaluated_arrays.push(variables.get(t).unwrap().clone()),
        }
    }

    evaluated_arrays
}


#[derive(Clone)]
/// What feeds `ndarray`s to the computation graph.
///
/// This is used to set `ndarray`'s array object to a `Placeholder` tensor.
/// Arbitrary number of inputs can be set to this object with builder-like usage.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let mut graph = ag::Graph::new();
/// let ref x: ag::Tensor = graph.placeholder();
/// let ref y: ag::Tensor = 3 * x;
///
/// // Fills placeholder `x`.
/// graph.feed(x, ndarray::arr1(&[2.]));
/// assert_eq!(6., graph.eval(&[y])[0][0]);
/// ```
pub struct Feed {
    pub hash_map: FnvHashMap<Tensor, NdArray>,
}

impl Feed {
    #[inline]
    pub fn new() -> Feed
    {
        Feed { hash_map: FnvHashMap::default() }
    }

    /// Adds a pair of `(Placeholder, A feed to the placeholder)` to the input object.
    #[inline]
    pub fn add<T>(mut self, placeholder: &Tensor, array: ndarray::Array<f32, T>) -> Self
    where
        T: ndarray::Dimension,
    {
        self.hash_map.insert(placeholder.clone(), array.into_dyn());
        // move self
        self
    }
}
