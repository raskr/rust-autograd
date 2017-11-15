extern crate fnv;
extern crate ndarray;

use self::fnv::FnvHashMap;
use ndarray_ext::NdArray;
use ops::dummy_op;
use std::rc::Rc;
use tensor;
use tensor::{RawTensor, Tensor};


#[derive(Clone)]
/// What is necessary to run computation graphs.
///
/// `Context` object is used:
/// - to create shared variable tensors
/// - to create placeholder tensors
/// - to run computation graphs actually
///
/// When you use a context object to evaluate a computation graph,
/// all the variables/placeholders in the graph must be what derives from
/// that context; otherwise it will panic.
///
/// # Examples
///
/// ```
/// extern crate ndarray;
/// extern crate autograd as ag;
///
/// // new Context
/// let mut ctx = ag::Context::new();
///
/// let ref x = ctx.placeholder();
/// let ref v = ctx.variable(ndarray::arr1(&[2., 2.]));
/// let ref b = ag::ones(&[2]);
/// let ref z = x + v + b;
///
/// // fills placeholder
/// ctx.feed(x, ndarray::arr1(&[1., 1.]));
///
/// // eval
/// assert_eq!(z.eval(&mut ctx).as_slice().unwrap(), &[4., 4.]);
/// ```
pub struct Context {
    pub variables: FnvHashMap<Tensor, NdArray>,
    pub outputs: FnvHashMap<Tensor, NdArray>,
}

impl Context {
    pub fn new() -> Context
    {
        Context {
            variables: FnvHashMap::default(),
            outputs: FnvHashMap::default(),
        }
    }

    #[inline]
    pub fn feed<T: ndarray::Dimension>(&mut self, placeholder: &Tensor, arr: ndarray::Array<f32, T>)
    {
        if "Placeholder" != placeholder.op.name() {
            panic!(
                "Don't call `Context::feed` with non placeholder, got: {}",
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
    /// let mut ctx = ag::Context::new();
    /// let ref x = ag::zeros(&[2, 2]);
    /// assert_eq!(ctx.eval(&[x])[0], ndarray::arr2(&[[0., 0.], [0., 0.]]).into_dyn())
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
    /// let mut ctx = ag::Context::new();
    /// let ref x: ag::Tensor = ctx.variable(ndarray::arr1(&[2.]));
    /// let ref y: ag::Tensor = 3 * x;
    ///
    /// assert_eq!(6., ctx.eval(&[y])[0][0]);
    /// assert!(ctx.variables.contains_key(x));
    /// assert_eq!(ctx.variables.get(x).unwrap(), &ndarray::arr1(&[2.]).into_dyn());
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
    /// The placeholder tensor is a dynamic input node to the computation Context,
    /// which can be filled on evaluation time.
    /// To fill the placeholders, use `Context::feed()`.
    ///
    /// # Examples
    ///
    /// ```
    /// extern crate ndarray;
    /// extern crate autograd as ag;
    ///
    /// let mut ctx = ag::Context::new();
    /// let ref x: ag::Tensor = ctx.placeholder();
    /// let ref y: ag::Tensor = 3 * x;
    ///
    /// // Fills placeholder `x`.
    /// ctx.feed(x, ndarray::arr1(&[2.]));
    /// assert_eq!(6., y.eval(&mut ctx)[0]);
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
    /// let mut ctx = ag::Context::new();
    /// let arr = ndarray::arr1(&[0., 0., 0.]);
    /// let ref con = ctx.constant(arr.clone());
    /// assert_eq!(arr.into_dyn(), ctx.eval(&[con])[0])
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
