extern crate fnv;
extern crate ndarray;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops::dummy_op;
use std::rc::Rc;
use tensor;
use tensor::{RawTensor, Tensor};


#[derive(Clone)]
/// Graph object.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// // new graph
/// let mut graph = ag::Graph::new();
///
/// let ref a = graph.placeholder();
/// let ref b = ag::ones(&[2]);
/// let ref v = graph.variable(ndarray::arr1(&[2., 2.]));
/// let ref z = a + b + v;
///
/// // fills placeholder
/// graph.feed(a, ndarray::arr1(&[1., 1.]));
///
/// // eval
/// assert_eq!(graph.eval(&[z])[0].as_slice().unwrap(), &[4., 4.]);
/// ```
pub struct Graph {
    pub variables: FnvHashMap<Tensor, NdArray>,
    pub outputs: FnvHashMap<Tensor, NdArray>,
}

impl Graph {
    pub fn new() -> Graph
    {
        Graph {
            variables: FnvHashMap::default(),
            outputs: FnvHashMap::default(),
        }
    }

    #[inline]
    pub fn feed<T: ndarray::Dimension>(&mut self, placeholder: &Tensor, arr: ndarray::Array<f32, T>)
    {
        if "Placeholder" != placeholder.op.name() {
            panic!(
                "Don't call `Graph::feed` with non placeholder, got: {}",
                placeholder.op.name()
            )
        }
        self.outputs.insert(placeholder.clone(), arr.into_dyn());
    }

    /// Evaluates input symbolic tensors.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    ///
    /// let mut graph = ag::Graph::new();
    ///
    /// let ref x = graph.constant(ag::ndarray_ext::standard_normal(&[4, 2]));
    /// let ref w = graph.variable(ag::ndarray_ext::standard_normal(&[2, 3]));
    /// let ref b = graph.variable(ag::ndarray_ext::zeros(&[1, 3]));
    /// let ref z = ag::matmul(x, w) + b;
    ///
    /// assert_eq!(graph.eval(&[z])[0].shape(), &[4, 3])
    /// ```
    pub fn eval(&mut self, xs: &[&Tensor]) -> Vec<ndarray::Array<f32, ndarray::IxDyn>>
    {
        let xs = xs.into_iter().map(|a| (*a).clone()).collect::<Vec<_>>();
        let ret = tensor::eval_tensors(xs.as_slice(), &mut self.variables, &mut self.outputs);
        self.outputs.clear();
        ret
    }


    /// Creates a shared variable tensor from rust-ndarray's array object.
    ///
    /// The shared variable behaves like any other tensors, except that
    /// it can be optimized with gradient descent methods
    /// implemented in `autograd::sgd`.
    /// For the usages, see https://github.com/perrier1034/rust-autograd/tree/master/examples
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut graph = ag::Graph::new();
    /// let ref x: ag::Tensor = graph.variable(ndarray::arr1(&[2.]));
    /// let ref y: ag::Tensor = 3 * x;
    ///
    /// assert_eq!(6., graph.eval(&[y])[0][0]);
    /// assert!(graph.variables.contains_key(x));
    /// assert_eq!(graph.variables.get(x).unwrap(), &ndarray::arr1(&[2.]).into_dyn());
    /// ```
    #[inline]
    pub fn variable<T>(&mut self, arr: ndarray::Array<f32, T>) -> Tensor
    where
        T: ndarray::Dimension,
    {
        let t = Tensor(Rc::new(RawTensor {
            op: Box::new(dummy_op::DummyOp { name: "Variable".to_string() }),
            inputs: vec![],
            top_rank: 0,
        }));
        self.variables.insert(t.clone(), arr.into_dyn());
        t
    }

    /// Creates a placeholder tensor.
    ///
    /// The placeholder tensor is a dynamic input node to the computation graph,
    /// which can be filled on evaluation time.
    /// To fill the placeholders, use `graph::feed()`.
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
    #[inline]
    pub fn placeholder(&self) -> Tensor
    {
        Tensor(Rc::new(RawTensor {
            op: Box::new(dummy_op::DummyOp { name: "Placeholder".to_string() }),
            inputs: vec![],
            top_rank: 0,
        }))
    }


    /// Creates a constant tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut graph = ag::Graph::new();
    /// let arr = ndarray::arr1(&[0., 0., 0.]);
    /// let ref con = graph.constant(arr.clone());
    /// assert_eq!(arr.into_dyn(), graph.eval(&[con])[0])
    /// ```
    #[inline]
    pub fn constant<T>(&mut self, arr: ndarray::Array<f32, T>) -> Tensor
    where
        T: ndarray::Dimension,
    {
        let t = Tensor(Rc::new(RawTensor {
            op: Box::new(dummy_op::DummyOp { name: "Constant".to_string() }),
            inputs: vec![],
            top_rank: 0,
        }));
        self.variables.insert(t.clone(), arr.into_dyn());
        t
    }
}
