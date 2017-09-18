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
use std::mem;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use topology;


/// Symbolic multi-dimensional array which supports
/// efficient gradient computation.
pub struct Tensor(pub Rc<RefCell<RawTensor>>);

pub struct RawTensor {
    // Operation of this node
    pub op: Box<ops::Op>,

    // Shared variable of this node; this is `Some` if created by:
    // - variable()
    // - constant()
    // - scalar()
    pub param: Option<NdArray>,

    // References to immediate predecessors.
    pub inputs: Vec<Tensor>,

    // rank number for topological ordering
    pub rank: usize,
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
        self.eval_with_input(Input::new())
    }

    #[inline]
    /// Returns a value of this node
    pub fn eval_with_input(&self, feed_dict: Input) -> ndarray::Array<f32, ndarray::IxDyn>
    {
        // pre process.
        // pack `feed_dict` in `memo` and collect shared variables
        let mut memo = feed_dict.hash_map;
        let mut variable_set = HashSet::new();
        self.visit_once(&mut |arg: &Tensor| {
            let mut cur_node = arg.borrow_mut();
            if let Some(v) = mem::replace(&mut cur_node.param, None) {
                variable_set.insert(arg.clone());
                memo.insert(arg.clone(), v);
            }
        });

        // run graph
        topology::perform_eval(self, &mut memo, false);

        //  make return value
        // TODO
        let result = if let Some(res) = memo.remove(self) {
            res
        } else {
            panic!("Some placeholders could'nt get initial value")
        };

        // post process.
        // return variable arrays to the original place
        for v in variable_set.into_iter() {
            mem::swap(&mut v.borrow_mut().param, &mut memo.remove(&v));
        }

        // return evaluated array
        result
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
    fn cmp(&self, other: &Self) -> Ordering
    {
        self.borrow().rank.cmp(&other.borrow().rank)
    }
}

impl PartialOrd for Tensor {
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
pub struct Input {
    pub hash_map: FnvHashMap<Tensor, NdArray>,
}

impl Input {
    #[inline]
    pub fn new() -> Input
    {
        Input { hash_map: FnvHashMap::default() }
    }

    #[inline]
    pub fn add<T>(mut self, symbolic_tensor: &Tensor, array: ndarray::Array<f32, T>) -> Self
    where
        T: ndarray::Dimension,
    {
        self.hash_map.insert(
            symbolic_tensor.clone(),
            array.into_dyn(),
        );
        // move self
        self
    }
}
