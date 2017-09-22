extern crate ndarray;
extern crate fnv;

use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::hash_set::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops;


/// Symbolic multi-dimensional array which supports
/// efficient gradient computation.
pub struct Tensor(pub Rc<RefCell<RawTensor>>);

pub struct RawTensor {
    /// Operation of this node
    pub op: Box<ops::Op>,

    /// Shared variable of this node;
    /// this is `Some` if created by `variable()`, `constant()` etc.
    pub param: Option<NdArray>,

    /// References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    /// rank number for topological ordering
    pub rank: usize,
}


#[inline]
pub fn eval_tensors(tensors: &[Tensor], feed_dict: Feed) -> Vec<NdArray>
{
    // move internal dict
    let mut memo = feed_dict.hash_map;

    // collects variables in the whole graph
    // and packs those in `memo`
    let mut vars = HashSet::<Tensor>::new();
    {
        let mut seen = HashSet::new();
        let mut stack = tensors.to_vec();
        // DFS
        while let Some(popped) = stack.pop() {
            seen.insert(popped.clone());
            // update stack
            for input in popped.borrow().inputs.iter() {
                if !seen.contains(input) {
                    stack.push(input.clone());
                }
            }
            // takes out `param` attr
            let param = mem::replace(&mut popped.borrow_mut().param, None);
            if let Some(v) = param {
                vars.insert(popped.clone());
                memo.insert(popped, v);
            }
        }
    }

    // run graph
    for t in tensors.iter() {
        ::topology::perform_eval(t, &mut memo, true);
    }

    // extracts target arrays
    let mut evaluated_arrays = Vec::with_capacity(tensors.len());
    for (i, t) in tensors.iter().enumerate() {
        // Need to handle cases where multiple gradient nodes
        // share an output array.
        // (Safe unwrapping is guaranteed by ::topology::symbolic_gradients())
        if tensors[i + 1..].contains(t) {
            // need to preserve the array for following nodes
            // => copy the array
            evaluated_arrays.push(memo.get(t).unwrap().clone());
        } else {
            // do not need to preserve
            // => move the array from memo
            evaluated_arrays.push(memo.remove(t).unwrap());
        }
    }

    // Don't forget to return param arrays to the original places
    for v in vars.iter() {
        mem::swap(&mut v.borrow_mut().param, &mut memo.remove(&v));
    }

    evaluated_arrays
}


impl Tensor {
    #[inline]
    pub fn is_source(&self) -> bool
    {
        self.borrow().inputs.is_empty()
    }

    /// Returns a value of this node
    #[inline]
    pub fn eval(&self) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        eval_tensors(&[self.clone()], Feed::new()).remove(0)
    }

    #[inline]
    /// Returns a value of this node
    pub fn eval_with_input(&self, feed_dict: Feed) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        eval_tensors(&[self.clone()], feed_dict).remove(0)
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

    pub fn feed_array<T: ndarray::Dimension>(&self, arr: ndarray::Array<f32, T>)
    {
        if self.borrow().op.name() != "Placeholder" {
            panic!("Can't feed array to non-placeholder");
        }
        self.borrow_mut().param = Some(arr.into_dyn());
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

#[derive(Clone)]
/// Dynamic input to the computation graph.
///
/// This is used to set `ndarray`'s array object to a `Placeholder` tensor.
/// Arbitrary number of inputs can be set to this object with builder-like usage.
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// let ref x = ag::placeholder();
/// let ref y = 3 * x;
///
/// // Fills placeholder `x`.
/// let feed_dict = ag::Feed::new().add(x, ndarray::arr1(&[2.]));
/// assert_eq!(6., y.eval_with_input(feed_dict)[0]);
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
