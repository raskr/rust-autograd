//! A collection of functions to manipulate `ag::Tensor` objects
use ndarray;

use crate::ndarray_ext::{ArrayRng, NdArray};
use crate::tensor::{AsTensor, Input, Tensor};
use crate::Float;
use rand::Rng;

mod activation_ops;
mod array_ops;
pub(crate) mod basic_source_ops;
pub(crate) mod binary_ops;
mod const_gen_ops;
mod conv_ops;
pub(crate) mod dot_ops;
pub mod gradient_descent_ops;
mod gradient_ops;
pub(crate) mod hook_ops;
mod math_ops;
mod mkl_ffi;
mod random_ops;
mod reduction_ops;
mod xent_ops;

// ---------------------------------------
// -- Ops to manipulate `Tensor` object --
// ---------------------------------------

impl<'graph, F: Float> Tensor<'graph, F> {
    /// Gets the `i` th float value of this tensor.
    ///
    /// Index `i` can be negative.
    ///
    /// ```
    /// use ndarray::{self, array};
    /// use autograd::{self as ag, tensor::Variable};
    ///
    /// ag::with(|g| {
    ///    let a = g.variable(array![[2., 3.], [4., 5.]]);
    ///    let b = a.access_elem(2);
    ///    assert_eq!(b.eval(&[]).unwrap()[ndarray::IxDyn(&[])], 4.);
    /// });
    /// ```
    pub fn access_elem(self, i: isize) -> Tensor<'graph, F> {
        let op = array_ops::IndexOp { index: i };
        Tensor::builder().append_input(&self).build(self.graph, op)
    }
}

impl<'graph, F: Float> crate::graph::Graph<F> {
    /// Symbolic gradient tensors of `xs` in the same order as `xs`'s
    ///
    /// # Arguments
    /// * `ys` - Targets of differentiation that are arbitrary shapes.
    /// * `xs` - Tensors with which differentiate `ys`.
    ///
    /// # Example
    /// Partial derivatives of `z = 2x^2 + 3y + 1`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let x = g.placeholder(&[]);
    ///     let y = g.placeholder(&[]);
    ///     let z = 2.*x*x + 3.*y + 1.;
    ///
    ///     // dz/dy
    ///     let gy = g.grad(&[z], &[y])[0];
    ///     // dz/dx
    ///     let gx = g.grad(&[z], &[x])[0];
    ///
    ///     // ddz/dx (differentiates `z` again)
    ///     let ggx = g.grad(&[gx], &[x])[0];
    ///
    ///     // evaluation of symbolic gradients
    ///     assert_eq!(3., gy.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    ///     assert_eq!(4., ggx.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    ///
    ///     // dz/dx requires to fill the placeholder `x`
    ///     assert_eq!(8., gx.eval(&[x.given(ndarray::arr0(2.).view())]).unwrap()[ndarray::IxDyn(&[])]);
    /// });
    /// ```
    pub fn grad<A, B>(&'graph self, ys_: &[A], xs: &[B]) -> Vec<Tensor<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let len = ys_.len();
        let mut ys = Vec::with_capacity(len);
        for y in ys_ {
            ys.push(self.reduce_sum_to_scalar(y));
        }
        let gys = vec![self.scalar(F::one()); len];
        unsafe { self.grad_with_default(&ys, xs, &gys) }
    }

    /// Computes `xs`'s gradients with `ys`'s already known gradients.
    ///
    /// Almost same spec as `grad`'s except that you can pass `ys`s already known gradients.
    /// If `ys_grads` are tensors filled with 1s, this function should be replaced with `grad`.
    ///
    /// NOTE: Please be careful to match `ys_grads[i].shape` and `ys[i].shape`,
    /// otherwise **undefined behavior** would happen.
    ///
    /// # Arguments
    /// * `ys` - Targets of differentiation.
    /// * `xs` - tensors with which differentiate `ys`.
    /// * `ys_grads` - Already known gradients of `ys`.
    ///
    /// # Returns
    /// Symbolic gradient tensors of `xs` in the same order as `xs`'graph.
    pub unsafe fn grad_with_default<A, B, C>(
        &'graph self,
        ys: &[A],
        xs: &[B],
        ys_grads: &[C],
    ) -> Vec<Tensor<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
        C: AsRef<Tensor<'graph, F>> + Copy,
    {
        let xs: Vec<_> = xs.iter().map(|x| x.as_ref().id).collect();
        let ys: Vec<_> = ys.iter().map(|y| y.as_ref().id).collect();
        let ys_grads: Vec<_> = ys_grads.iter().map(|x| x.as_ref().id).collect();
        crate::gradient::symbolic_gradients(ys.as_slice(), xs.as_slice(), ys_grads.as_slice(), self)
    }

    /// Computes jacobians for variables.
    ///
    /// # Arguments
    /// * `y` - Target of differentiation.
    /// * `xs` - Tensors with which differentiate `ys`.
    /// * `y_size` - (flattened) size of `y`
    ///
    /// # Returns
    /// Jacobians for each variable. Each one is a matrix of shape `(y_size, x size)`.
    ///
    /// Note: the current implementation works correctly but is unoptimized for serious use.
    ///
    /// ```
    /// use autograd as ag;
    /// use ag::tensor::Variable;
    ///
    /// ag::with(|g| {
    ///    let rng = ag::ndarray_ext::ArrayRng::<f32>::default();
    ///    let a = g.variable(rng.standard_normal(&[4, 2]));
    ///    let b = g.variable(rng.standard_normal(&[2, 3]));
    ///    let c = g.matmul(a, b);
    ///    let j = g.jacobians(c, &[a, b], 4*3);
    ///
    ///    assert_eq!(j[0].eval(&[]).unwrap().shape(), &[4*3, 4*2]);
    ///    assert_eq!(j[1].eval(&[]).unwrap().shape(), &[4*3, 2*3]);
    /// });
    /// ```
    pub fn jacobians<A, B>(
        &'graph self,
        y_: A,
        xs_: &[B],
        objective_len: usize,
    ) -> Vec<Tensor<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let y = y_.as_ref();
        // let xs: Vec<_> = xs_.iter().map(|x| x.as_ref().inner()).collect();
        let mut vec_vec = Vec::with_capacity(objective_len);
        // let gy = self.scalar(F::one()).inner();
        for i in 0..objective_len as isize {
            vec_vec.push(self.grad(&[y.access_elem(i)], xs_));
        }

        let len = xs_.len();
        let mut ret = Vec::with_capacity(len);
        // post process gradients
        for i in 0..len {
            // jac is matrix
            let mut jac = Vec::with_capacity(objective_len);
            for vec in &vec_vec {
                jac.push(self.expand_dims(self.flatten(&vec[i]), &[0]));
            }
            // (y size, x size)
            ret.push(self.concat(&jac, 0));
        }
        ret
    }

    /// (Experimental) Computes hessian vector product
    pub fn _hessian_vector_product<A, B, C>(
        &'graph self,
        ys: &[A],
        xs: &[B],
        vectors: &[C],
    ) -> Vec<Tensor<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
        C: AsRef<Tensor<'graph, F>> + Copy,
    {
        let grads = self.grad(ys, xs);
        let products = grads
            .into_iter()
            .zip(vectors)
            .map(|(g, &v)| *g.as_ref() * *v.as_ref())
            .collect::<Vec<_>>();
        self.grad(products.as_slice(), xs)
    }

    /// Stops gradient propagation.
    ///
    /// Guarantees that the gradient is not propagated to the tensors behind this
    /// during gradient computation.
    pub fn stop_gradient<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_differentiable(false)
            .build(self, gradient_ops::StopGradient)
    }

    /// Creates a placeholder tensor.
    ///
    /// Behaves like TensorFlow's placeholder object.
    /// `shape_[i]` must be a positive value, or -1 which means dynamic dim.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///     let x = g.placeholder(&[2]);
    ///
    ///     // Fills placeholder, then eval
    ///     let arr = ndarray::array![1., 1.].into_dyn();
    ///     assert_eq!(x.eval(&[x.given(arr.view())]), Ok(arr));
    /// });
    /// ```
    #[inline]
    pub fn placeholder(&'graph self, shape_: &[isize]) -> Tensor<'graph, F> {
        let b = Tensor::builder().set_is_placeholder(true);
        let rank = shape_.len();
        let b = if rank == 0 || -1 != shape_[0] {
            b.set_shape(
                &self.convert_to_tensor(
                    NdArray::from_shape_vec(
                        ndarray::IxDyn(&[rank]),
                        shape_
                            .iter()
                            .map(|&x| F::from(x).unwrap())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap(),
                ),
            )
        } else {
            b
        };
        let b = b.set_known_shape(shape_.to_vec());
        b.build(self, basic_source_ops::Placeholder)
    }

    /// Returns a `Tensor` representation of the input tensor's shape
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let x: ag::Tensor<f32> = g.zeros(&[2, 3]);
    ///    let s = g.shape(x);
    ///    assert_eq!(&[2., 3.], s.eval(&[]).unwrap().as_slice().unwrap());
    /// });
    /// ```
    pub fn shape<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        unsafe {
            if let Some(id) = x.as_ref().inner().shape {
                self.tensor(id)
            } else {
                Tensor::builder()
                    .append_input(x.as_ref())
                    .set_differentiable(false)
                    .build(self, array_ops::Shape)
            }
        }
    }

    /// Returns the (symbolic) size of the input tensor
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[4, 3]);
    ///    let b = g.size(a);
    ///
    ///    assert_eq!(12., b.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// });
    /// ```
    pub fn size<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_differentiable(false)
            .build(self, array_ops::Size)
    }

    /// Returns the (symbolic) rank of the input tensor
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let x: ag::Tensor<f32> = g.zeros(&[2, 3, 4]);
    ///    let r = g.rank(x);
    ///    assert_eq!(3., r.eval(&[]).unwrap()[ndarray::IxDyn(&[])]);
    /// });
    /// ```
    pub fn rank<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_differentiable(false)
            .build(self, array_ops::Rank)
    }

    /// Elementwise sine
    pub fn sin<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Sin)
    }

    /// Elementwise cosine
    pub fn cos<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Cos)
    }

    /// Elementwise tangent
    pub fn tan<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Tan)
    }

    /// Elementwise arcsin
    pub fn asin<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Asin)
    }

    /// Elementwise arccos
    pub fn acos<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Acos)
    }

    /// Elementwise arctan
    pub fn atan<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Atan)
    }

    /// Elementwise hyperbolic sine
    pub fn sinh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Sinh)
    }

    /// Elementwise hyperbolic cosine
    pub fn cosh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Cosh)
    }

    /// Elementwise hyperbolic tangent
    pub fn tanh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Tanh)
    }

    /// Elementwise hyperbolic arcsin
    pub fn asinh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Asinh)
    }

    /// Elementwise hyperbolic arccos
    pub fn acosh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Acosh)
    }

    /// Elementwise hyperbolic arctan
    pub fn atanh<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Atanh)
    }

    #[doc(hidden)]
    /// Gets n th tensor in `x`.
    ///
    /// `x` must be a result of a multi-outputs op;
    /// otherwise index-out-of-bounds error may happen.
    pub fn nth_tensor<A>(&'graph self, x: A, n: usize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_input_indices(&[n])
            .build(self, activation_ops::Identity)
    }

    /// Identity function without copy.
    pub fn identity<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, activation_ops::Identity)
    }

    #[inline]
    fn infer_bin_op_shape<A, B>(&'graph self, shape_a: A, shape_b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[shape_a.as_ref(), shape_b.as_ref()])
            .build(self, array_ops::InferBinOpShape)
    }

    /// Elementwise addition.
    ///
    /// This can be replaced with `+` operation of Tensor.
    #[inline]
    pub fn add<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.infer_bin_op_shape(self.shape(a), self.shape(b)))
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, binary_ops::AddOp)
    }

    /// Element-wise subtraction.
    ///
    /// This can be replaced with `-` operation of Tensor.
    #[inline]
    pub fn sub<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.infer_bin_op_shape(self.shape(a), self.shape(b)))
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, binary_ops::SubOp)
    }

    /// Elementwise multiplication.
    ///
    /// This can be replaced with `*` operation of Tensor.
    #[inline]
    pub fn mul<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.infer_bin_op_shape(self.shape(a), self.shape(b)))
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, binary_ops::MulOp)
    }

    /// Elementwise division.
    ///
    /// This can be replaced with `/` operation of Tensor.
    #[inline]
    pub fn div<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.infer_bin_op_shape(self.shape(a), self.shape(b)))
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, binary_ops::DivOp)
    }

    /// Elementwise sqrt
    #[inline]
    pub fn sqrt<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Sqrt)
    }

    /// Elementwise pow
    pub fn pow<A>(&'graph self, x: A, a: F) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Pow { a })
    }

    /// Elementwise base e (napier) logarithm
    pub fn ln<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Ln)
    }

    /// Elementwise base 2 logarithm
    pub fn log2<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Log2)
    }

    /// Elementwise base 10 logarithm
    pub fn log10<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Log10)
    }

    /// Elementwise base e (napier) exponential
    pub fn exp<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Exp)
    }

    /// Elementwise base 2 exponential
    pub fn exp2<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Exp2)
    }

    /// Elementwise base 10 exponential
    pub fn exp10<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .set_shape(&self.shape(x))
            .build(self, math_ops::Exp10)
    }

    /// Returns the max of x and y (i.e. x > y ? x : y) element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![1., 2., 3.]);
    ///    let b = g.constant(array![3., 2., 1.]);
    ///    let c = g.maximum(a, b);
    ///    assert_eq!(c.eval(&[]), Ok(array![3., 2., 3.].into_dyn()));
    /// });
    /// ```
    pub fn maximum<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Maximum)
    }

    /// Returns the min of x and y (i.e. x > y ? y : x) element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![1., 2., 3.]);
    ///    let b = g.constant(array![3., 2., 1.]);
    ///    let c = g.minimum(a, b);
    ///    assert_eq!(c.eval(&[]), Ok(array![1., 2., 1.].into_dyn()));
    /// });
    /// ```
    pub fn minimum<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Minimum)
    }

    /// Adds all input tensors, element-wise.
    ///
    /// All the input tensors must have same shapes.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.ones(&[2, 2]);
    ///    let b = g.ones(&[2, 2]);
    ///    let c = g.ones(&[2, 2]);
    ///    let d = g.add_n(&[a, b, c]);
    ///
    ///    assert_eq!(d.eval(&[]).unwrap().shape(), &[2, 2]);
    ///    assert_eq!(d.eval(&[]), Ok(array![[3., 3.], [3., 3.]].into_dyn()));
    /// });
    /// ```
    pub fn add_n<A>(&'graph self, xs: &[A]) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let len = xs.len();
        assert_ne!(len, 0);
        if len == 1 {
            *xs[0].as_ref()
        } else {
            let _xs = xs.iter().map(|x| x.as_ref()).collect::<Vec<_>>();
            Tensor::builder()
                .set_ro_inputs(_xs.as_slice())
                .set_shape(&self.shape(xs[0]))
                .build(self, array_ops::AddN)
        }
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// if `a[i] == b[i]` then `return-value[i]` will be 1 else 0
    ///
    /// # Panics
    /// When broadcast is impossible
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![1., 2., 3.]);
    ///    let b = g.constant(array![3., 2., 1.]);
    ///    let c = g.equal(a, b);
    ///    assert_eq!(c.eval(&[]), Ok(ndarray::arr1(&[0., 1., 0.]).into_dyn()));
    /// });
    /// ```
    pub fn equal<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Equal)
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// if `a[i] != b[i]` then `return-value[i]` will be 1 else 0
    ///
    /// # Panics
    /// When broadcast is impossible
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![1., 2., 3.]);
    ///    let b = g.constant(array![3., 2., 1.]);
    ///    let c = g.not_equal(a, b);
    ///    assert_eq!(c.eval(&[]), Ok(array![1., 0., 1.].into_dyn()));
    /// });
    /// ```
    pub fn not_equal<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::NotEqual)
    }

    /// Takes argmin along specified axis.
    ///
    /// `axis` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[3., 4.], [6., 5.]]);
    ///    let y = g.argmin(x, 1, false);
    ///
    ///    assert_eq!(y.eval(&[]), Ok(array![0., 1.].into_dyn()));
    /// });
    /// ```
    pub fn argmin<A>(&'graph self, x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = reduction_ops::ArgMin { axis, keep_dim };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Takes argmax along specified axis.
    ///
    /// `axis` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[3., 4.], [6., 5.]]);
    ///    let y = g.argmax(x, 1, false);
    ///
    ///    assert_eq!(y.eval(&[]), Ok(array![1., 0.].into_dyn()));
    /// });
    /// ```
    pub fn argmax<A>(&'graph self, x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = reduction_ops::ArgMax { axis, keep_dim };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Expands the shape (inserts axes).
    ///
    /// Each axis can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[3]);
    ///    let b = g.expand_dims(a, &[0, 2]);
    ///    assert_eq!(b.eval(&[]).unwrap().shape(), &[1, 3, 1]);
    /// });
    /// ```
    pub fn expand_dims<A, AT>(&'graph self, x: A, axes: &AT) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, array_ops::ExpandDims)
    }

    /// Remove the specified dims.
    ///
    /// Each axis can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[1, 3, 1]);
    ///    let b = g.squeeze(a, &[0, 2]);
    ///    assert_eq!(b.eval(&[]).unwrap().shape(), &[3]);
    /// })
    /// ```
    pub fn squeeze<A, AT>(&'graph self, x: A, axes: &AT) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, array_ops::Squeeze)
    }

    /// Tiles the input tensor along specified axis.
    ///
    /// Tiles input tensor `num` times along `axis`.
    /// `axis` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 2.], [3., 3.]]);
    ///    let y = g.tile(x, 0, 2);
    ///
    ///    assert_eq!(
    ///        y.eval(&[]),
    ///        Ok(array![[2., 2.], [3., 3.], [2., 2.], [3., 3.]].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn tile<A>(&'graph self, x: A, axis: isize, num: usize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = array_ops::Tile { axis, num };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Limits all elements of `x` so as to be within `[min, max]`
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![2., 4., 6.]);
    ///    let y = g.clip(x, 3., 5.);
    ///    assert_eq!(y.eval(&[]), Ok(ndarray::arr1(&[3., 4., 5.]).into_dyn()));
    /// });
    /// ```
    pub fn clip<A>(&'graph self, x: A, min: F, max: F) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = array_ops::Clip { min, max };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Takes max along specified axes.
    ///
    /// Each of element of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_max(&x, &[0], false);
    ///    assert_eq!(y.eval(&[]), Ok(array![3., 4.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_max<A, AT>(&'graph self, x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = reduction_ops::ReduceMax {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes min along specified axes.
    ///
    /// Each of element of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_min(&x, &[0], false);
    ///    assert_eq!(y.eval(&[]), Ok(array![2., 1.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_min<A, AT>(&'graph self, x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = reduction_ops::ReduceMin {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Sum up all the elements to a scalar value (0-D Tensor).
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_sum_to_scalar(&x);
    ///    assert_eq!(y.eval(&[]), Ok(ndarray::arr0(10.).into_dyn()));
    /// });
    /// ```
    pub fn reduce_sum_to_scalar<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(x.as_ref())
            .build(self, reduction_ops::ReduceSumToScalar)
    }

    /// Takes sumation along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_sum(&x, &[1], false);
    ///
    ///    assert_eq!(y.eval(&[]), Ok(array![6., 4.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_sum<A, AT>(&'graph self, x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = reduction_ops::ReduceSum {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes mean along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_mean(x, &[1], false);
    ///    assert_eq!(y.eval(&[]), Ok(array![3., 2.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_mean<A, AT>(&'graph self, x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = reduction_ops::ReduceMean {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Takes product along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[2., 4.], [3., 1.]]);
    ///    let y = g.reduce_prod(&x, &[1], false);
    ///    assert_eq!(y.eval(&[]), Ok(array![8., 3.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_prod<A, AT>(&'graph self, x: A, axes: &AT, keep_dims: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = reduction_ops::ReduceProd {
            keep_dims,
            sparse_axes: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Compute population variance along specified axes.
    ///
    /// Elements of `axes` can be negative.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let x = g.constant(array![[1., 1.], [2., 2.]]);
    ///    let y = g.reduce_variance(&x, &[1], false);
    ///    assert_eq!(y.eval(&[]), Ok(array![0., 0.].into_dyn()));
    /// });
    /// ```
    pub fn reduce_variance<A, AT>(
        &'graph self,
        x: A,
        axes: &AT,
        keep_dims: bool,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let x = x.as_ref();
        self.reduce_mean(
            self.square(x - self.reduce_mean(x, axes, true)),
            axes,
            keep_dims,
        )
    }

    /// Reshapes the input tensor without copy.
    ///
    /// Only one element in `shape` can be `-1`.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let x: ag::Tensor<f32> = g.zeros(&[3, 2, 2]);
    ///    let y = g.reshape(&x, &[3, -1]);
    ///    assert_eq!(y.eval(&[]), Ok(ag::ndarray_ext::zeros::<f32>(&[3, 4])));
    /// });
    /// ```
    pub fn reshape<A, AT>(&'graph self, x: A, shape: &AT) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &shape.as_tensor(self)])
            .build(self, array_ops::Reshape)
    }

    /// Flattens the input tensor into 1-ranked (vector) without copy.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let x: ag::Tensor<f32> = g.zeros(&[3, 2, 2]);
    ///    let z = g.flatten(x);
    ///    assert_eq!(z.eval(&[]).unwrap().shape(), &[12]);
    /// });
    /// ```
    pub fn flatten<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &self.scalar(F::one().neg())])
            .set_shape(&self.shape(x))
            .build(self, array_ops::Reshape)
    }

    /// Returns -1 if x < 0, 0 if x==0, 1 if x > 0, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![-5., 4.5, 0.]);
    ///    let b = g.sign(a);
    ///    assert_eq!(
    ///        b.eval(&[]).unwrap().as_slice().unwrap(),
    ///        &[-1., 1., 0.]
    ///    );
    /// });
    /// ```
    pub fn sign<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .append_input(a.as_ref())
            .build(self, math_ops::Sign)
    }

    /// Returns the largest integer less than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![-0.2, 0., 0.2]);
    ///    let b = g.abs(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(ndarray::arr1(&[0.2, 0., 0.2]).into_dyn())
    ///    );
    /// });
    /// ```
    pub fn abs<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .append_input(a.as_ref())
            .build(self, math_ops::Abs)
    }

    /// Returns the largest integer less than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]);
    ///    let b = g.floor(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![-2., -2., -1.,  0.,  1.,  1.,  2.].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn floor<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .append_input(a.as_ref())
            .build(self, math_ops::Floor)
    }

    /// Performs the `-` operation.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![2., 3.]);
    ///    let b = g.neg(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![-2., -3.].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn neg<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .append_input(a.as_ref())
            .build(self, math_ops::NegOp)
    }

    /// Takes square of the input.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![2., 3.]);
    ///    let b = g.square(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![4., 9.].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn square<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .append_input(a.as_ref())
            .build(self, math_ops::Square)
    }

    /// Returns the `1/x`, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![2.]);
    ///    let b = g.inv(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![0.5].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn inv<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, math_ops::Inv)
    }

    /// Returns the `1/sqrt(x)`, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![4.]);
    ///    let b = g.inv_sqrt(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![0.5].into_dyn())
    ///    );
    /// });
    /// ```
    pub fn inv_sqrt<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, math_ops::InvSqrt)
    }

    /// Returns the smallest integer greater than or equal to a number, element-wise.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0]);
    ///    let b = g.ceil(a);
    ///    assert_eq!(
    ///        b.eval(&[]),
    ///        Ok(array![-1., -1., -0.,  1.,  2.,  2.,  2.].into_dyn())
    ///    );
    ///
    /// });
    /// ```
    pub fn ceil<A>(&'graph self, a: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(a))
            .append_input(a.as_ref())
            .build(self, math_ops::Ceil)
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Greater)
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn greater_equal<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::GreaterEqual)
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::Lesser)
    }

    /// Compares a couple of tensors and returns a binary tensor.
    ///
    /// # Panics
    /// When broadcast is impossible
    pub fn lesser_equal<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, math_ops::LesserEqual)
    }

    /// Elementwise logistic sigmoid function.
    pub fn sigmoid<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, activation_ops::Sigmoid)
    }

    /// Elementwise exponential linear unit.
    ///
    /// See https://arxiv.org/abs/1511.07289
    pub fn elu<A>(&'graph self, x: A, alpha: F) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, activation_ops::ELU { alpha })
    }

    /// Elementwise rectified linear unit.
    pub fn relu<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, activation_ops::ReLU)
    }

    /// Elementwise leaky relu.
    ///
    /// In common, `alpha` is around 0.1 ~ 0.2.
    ///
    /// See http://web.stanford.edu/~awni/papers/relu_hybrid_icml2013_final.pdf
    pub fn leaky_relu<A>(&'graph self, x: A, alpha: F) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        self.maximum(x, self.scalar(alpha) * *x.as_ref())
    }

    /// Elementwise softplus.
    pub fn softplus<A>(&'graph self, x: A) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, activation_ops::Softplus)
    }

    /// Computes `log(sum(exp(x)))` along specified axis.
    ///
    /// `axis` can be negative.
    pub fn reduce_logsumexp<A>(&'graph self, x: A, axis: isize, keep_dim: bool) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>>,
    {
        let op = math_ops::LogSumExp {
            axis,
            keep_dims: keep_dim,
        };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Log softmax function.
    ///
    /// Computes `softmax(x)` along specified axis and
    /// takes logarithm of it.
    /// `axis` can be negative.
    pub fn log_softmax<A>(&'graph self, x: A, axis: isize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_shape(&self.shape(x))
            .append_input(x.as_ref())
            .build(self, xent_ops::LogSoftmax { axis })
    }

    /// Computes softmax along specified axis
    ///
    /// `axis` can be negative.
    pub fn softmax<A>(&'graph self, x: A, axis: isize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = activation_ops::Softmax { axis };
        Tensor::builder().append_input(x.as_ref()).build(self, op)
    }

    /// Computes `binary_cross_entropy(sigmoid(y), t)`.
    ///
    /// This function is better than that combination in that it can prevent
    /// underflow of `log(sigmoid)`.
    ///
    /// # Arguments
    /// * `y` - Tensor with arbitrary shape
    /// * `t` - Ground-truth Tensor with same shape as `y`'graph
    ///
    /// # Panics
    /// When y.shape != t.shape.
    ///
    /// # Returns
    /// Loss tensor with same shape as inputs's shapes
    pub fn sigmoid_cross_entropy<A, B>(&'graph self, y: A, t: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = xent_ops::SigmoidCrossEntropy;
        Tensor::builder()
            .set_shape(&self.shape(y))
            .set_ro_inputs(&[y.as_ref(), t.as_ref()])
            .build(self, op)
    }

    /// Computes `categorical_cross_entropy(softmax(y), t)`.
    ///
    /// This function is better than that combination in that it can prevent
    /// underflow of `log(softmax)`.
    ///
    /// # Arguments
    /// * `y` - Tensor with shape (batch_size, num_classes)
    /// * `t` - Tensor with shape (batch_size, num_classes)
    ///
    /// # Returns
    /// Loss tensor with shape (batch_size, 1)
    pub fn softmax_cross_entropy<A, B>(&'graph self, y: A, t: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = xent_ops::SoftmaxCrossEntropy;
        Tensor::builder()
            .set_ro_inputs(&[y.as_ref(), t.as_ref()])
            .build(self, op)
    }

    /// A variant of `softmax_cross_entropy`.
    ///
    /// The behavior of this function is same as `softmax_cross_entropy`
    /// except that `t` is **not** batch of one-hot distributions but batch of ground truth label ids.
    ///
    /// # Arguments
    /// * `y` - Tensor with shape (batch_size, num_classes)
    /// * `t` - Tensor with shape (batch_size,) or (batch_size, 1)
    ///
    /// # Returns
    /// Loss tensor with shape (batch_size, 1)
    pub fn sparse_softmax_cross_entropy<A, B>(&'graph self, y: A, t: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = xent_ops::SparseSoftmaxCrossEntropy;
        Tensor::builder()
            .set_ro_inputs(&[y.as_ref(), t.as_ref()])
            .build(self, op)
    }

    /// Matrix multiplication.
    ///
    /// Both `a` and `b` must be 2-ranked tensors.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[4, 2]);
    ///    let b: ag::Tensor<f32> = g.zeros(&[2, 3]);
    ///    let c = g.matmul(a, b);
    ///    assert_eq!(c.eval(&[]).unwrap().shape(), &[4, 3]);
    /// });
    /// ```
    ///
    /// This function supports only f32 and f64.
    pub fn matmul<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(
                self,
                dot_ops::MatMul {
                    transpose_a: false,
                    transpose_b: false,
                },
            )
    }

    /// Computes tensor-dot-product (tensor contraction) along specified axes.
    ///
    /// # Arguments
    /// * `a` - First input tensor
    /// * `b` - Second input tensor
    /// * `a_axes` - `a`'s Contraction axes
    /// * `b_axes` - `b`'s Contraction axes
    ///
    /// NOTE:
    ///
    /// * length of `a_axes` and `b_axes` must match.
    /// * Each axis number can be negative.
    /// * Supports only f32 and f64.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[3, 4, 5]);
    ///    let b: ag::Tensor<f32> = g.zeros(&[4, 3, 2]);
    ///    let c = g.tensordot(a, b, &[1, 0], &[0, 1]);
    ///    assert_eq!(c.eval(&[]).unwrap().shape(), &[5, 2]);
    /// });
    /// ```
    ///
    /// For detailed description,
    /// see https://docs.scipy.org/doc/numpy/reference/generated/numpy.tensordot.html.
    pub fn tensordot<A, B, AT1, AT2>(
        &'graph self,
        a: A,
        b: B,
        a_axes: &AT1,
        b_axes: &AT2,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
        AT1: AsTensor<'graph, F>,
        AT2: AsTensor<'graph, F>,
    {
        // Preprocess
        let pre = &Tensor::builder()
            .set_ro_inputs(&[
                a.as_ref(),
                b.as_ref(),
                &a_axes.as_tensor(self),
                &b_axes.as_tensor(self),
            ])
            .build(self, dot_ops::TensordotPreprocess);
        let final_shape = self.nth_tensor(pre, 0);
        let perm_a = self.nth_tensor(pre, 1);
        let perm_b = self.nth_tensor(pre, 2);
        let new_shape_a = self.nth_tensor(pre, 3);
        let new_shape_b = self.nth_tensor(pre, 4);

        let a_reshaped = self.reshape(self.transpose(a, &perm_a), &new_shape_a);
        let b_reshaped = self.reshape(self.transpose(b, &perm_b), &new_shape_b);

        // matmul
        let mm = self.matmul(a_reshaped, b_reshaped);
        self.reshape(mm, &final_shape)
    }

    /// Batched matrix multiplication with inputs's transposition.
    ///
    /// The rank of `a` and `b` must be equals.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[2, 3, 2, 4]);
    ///    let b: ag::Tensor<f32> = g.zeros(&[2, 3, 2, 3]);
    ///    let c = g.batch_matmul_t(a, b, true, false);
    ///    assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
    /// });
    /// ```
    ///
    /// This function supports only f32 and f64.
    /// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
    pub fn batch_matmul_t<A, B>(
        &'graph self,
        a: A,
        b: B,
        trans_a: bool,
        trans_b: bool,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = dot_ops::BatchMatMul {
            transpose_a: trans_a,
            transpose_b: trans_b,
        };
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, op)
    }

    /// Batched matrix multiplication.
    ///
    /// The rank of `a` and `b` must be equals.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.ones(&[2, 3, 4, 2]);
    ///    let b: ag::Tensor<f32> = g.ones(&[2, 3, 2, 3]);
    ///    let c = g.batch_matmul(a, b);
    ///    assert_eq!(c.eval(&[]).unwrap().shape(), &[2, 3, 4, 3]);
    /// });
    /// ```
    ///
    /// This function supports only f32 and f64.
    /// For detailed description, see https://www.tensorflow.org/api_docs/python/tf/matmul
    pub fn batch_matmul<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = dot_ops::BatchMatMul {
            transpose_a: false,
            transpose_b: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, op)
    }

    /// Takes diff between two tensors.
    ///
    /// Returns the sorted, unique values in `a` that are not in `b`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let a = g.constant(array![4., 1., 5., 2., 3., 6.]);
    ///    let b = g.constant(array![[2., 3.], [1., 4.]]);
    ///    let c = g.setdiff1d(a, b);
    ///    assert_eq!(
    ///        c.eval(&[]),
    ///        Ok(ndarray::arr1(&[5., 6.]).into_dyn())
    ///    )
    /// });
    /// ```
    ///
    pub fn setdiff1d<A, B>(&'graph self, a: A, b: B) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = array_ops::SetDiff1D;
        Tensor::builder()
            .set_ro_inputs(&[a.as_ref(), b.as_ref()])
            .build(self, op)
    }

    /// Permutes dimensions without copy.
    ///
    /// It's like TensorFlow or NumPy's.
    /// `x`'s rank (ndim) and `axes.len()` must match.
    ///
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[1, 2, 3, 4, 5]);
    ///    let b = g.transpose(a, &[4, 2, 3, 0, 1]);
    ///    assert_eq!(b.eval(&[]).unwrap().shape(), &[5, 3, 4, 1, 2]);
    /// });
    /// ```
    pub fn transpose<A, AT>(&'graph self, x: A, axes: &AT) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let op = math_ops::Transpose { invert_axes: false };
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), &axes.as_tensor(self)])
            .build(self, op)
    }

    /// Splits input tensors into parts.
    ///
    /// Splits `x` into `sizes.len()` parts along `axis`.
    ///
    /// The size of dimension of each part is `sizes[i]` on `axis`, but is
    /// `x.shape[i]` on other axis (similar to TensorFlow's `split`).
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[3, 7, 5]);
    ///    let b = g.split(a, &[2, 3, 2], 1);
    ///
    ///    let evaluated = g.eval(&[&b[0], &b[1], &b[2]], &[]);
    ///    let e0 = &evaluated[0];
    ///    let e1 = &evaluated[1];
    ///    let e2 = &evaluated[2];
    ///
    ///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 2, 5]);
    ///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 3, 5]);
    ///    assert_eq!(e2.as_ref().unwrap().shape(), &[3, 2, 5]);
    /// });
    /// ```
    pub fn split<A>(&'graph self, x: A, sizes: &[usize], axis: isize) -> Vec<Tensor<'graph, F>>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        let len = sizes.len();
        let mut ret = Vec::with_capacity(len);
        for i in 0..len {
            let mut start_index = 0usize;
            for &size in sizes[..i].iter() {
                start_index += size;
            }
            let end_index = start_index + sizes[i];
            ret.push(Tensor::builder().append_input(x.as_ref()).build(
                self,
                array_ops::Split {
                    start_index: start_index as isize,
                    end_index: end_index as isize,
                    axis,
                },
            ));
        }
        ret
    }

    /// Slices the input tensor.
    ///
    /// # Arguments
    /// * `x` - Tensor with arbitrary shape.
    /// * `starts` - Inclusive start indices for the dimensions.
    /// * `ends` - End indices for the dimensions. **Each index is inclusive if it is negative and exclusive if it's not.**
    ///
    /// NOTE: Negative values in `starts` and `ends` are counted from the back of the axis.
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[4, 4]);
    ///    let b = g.slice(a, &[0, 0], &[-1, 2]); // numpy equivalent is a[:, 0:2]
    ///
    ///    assert_eq!(b.eval(&[]).unwrap().shape(), &[4, 2]);
    /// });
    /// ```
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[4, 4]);
    ///    let b = g.slice(a, &[0, 0], &[-2, 2]); // numpy equivalent is a[:-1, :2]
    ///
    ///    assert_eq!(b.eval(&[]).unwrap().shape(), &[3, 2]);
    /// });
    /// ```
    pub fn slice<A>(&'graph self, x: A, starts: &[isize], ends: &[isize]) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        // TODO: Make starts and ends ArrayLike
        assert_eq!(starts.len(), ends.len());
        let starts_ends = starts.iter().zip(ends.iter());

        let indices = starts_ends
            .map(|(s, &e)| {
                let e = if e == -1 {
                    None
                } else {
                    Some(if e < -1 { e + 1 } else { e })
                };
                let slice = ndarray::Slice::new(*s, e, 1);
                ndarray::SliceOrIndex::from(slice)
            })
            .collect::<Vec<ndarray::SliceOrIndex>>();

        Tensor::builder()
            .append_input(x.as_ref())
            .build(self, array_ops::Slice { indices })
    }

    /// Concatenates input tensors along specified axis.
    ///
    /// `axis` can be negative.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[3, 2]);
    ///    let b: ag::Tensor<f32> = g.zeros(&[3, 2]);
    ///    let c: ag::Tensor<f32> = g.zeros(&[3, 2]);
    ///    let d = g.concat(&[a, b, c], 0);
    ///
    ///    assert_eq!(d.eval(&[]).unwrap().shape(), &[9, 2]);
    /// });
    /// ```
    pub fn concat<A>(&'graph self, tensors: &[A], axis: isize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        assert_ne!(tensors.len(), 0);
        let op = array_ops::Concat { axis };
        let vec: Vec<_> = tensors.iter().map(|t| Input::new(t.as_ref())).collect();
        Tensor::builder().set_inputs_vec(vec).build(self, op)
    }

    /// Gathers subviews from the input tensor.
    ///
    /// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
    /// For example, this can be used for embedding vectors lookup etc.
    ///
    /// Unlike `ag::gather`, `indices` can contain negative elements.
    ///
    /// # Returns
    /// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let param = g.zeros(&[5, 4, 8, 2]);
    ///    let indices = g.constant(array![[5., -1., 3.], [2., 1., -2.]]);
    ///    let y = g.gather_common(param, indices, 2);
    ///
    ///    assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])
    /// });
    /// ```
    pub fn gather_common<A, B>(&'graph self, param: A, indices: B, axis: isize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: true,
        };
        Tensor::builder()
            .set_ro_inputs(&[indices.as_ref(), param.as_ref()])
            .build(self, op)
    }

    /// Gathers subviews from the input tensor.
    ///
    /// Same spec as https://www.tensorflow.org/api_docs/python/tf/gather.
    /// For example, this can be used for embedding vectors lookup etc.
    ///
    /// # Returns
    /// Tensor with shape `param.shape[..axis] + indices.shape + param.shape[axis+1..]`
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    /// use ag::tensor::Constant;
    ///
    /// ag::with(|g| {
    ///    let param = g.zeros(&[5, 4, 8, 2]);
    ///    let indices = g.constant(array![[5., 4., 3.], [2., 1., 0.]]);  // shape: (2, 3)
    ///    let y = g.gather(param, indices, 2);
    ///
    ///    assert_eq!(y.eval(&[]).unwrap().shape(), &[5, 4, 2, 3, 2])
    /// });
    /// ```
    pub fn gather<A, B>(&'graph self, param: A, indices: B, axis: isize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        let op = array_ops::Gather {
            axis,
            should_normalize_negative_indices: false,
        };
        Tensor::builder()
            .set_ro_inputs(&[indices.as_ref(), param.as_ref()])
            .build(self, op)
    }

    /// Normalizes the input tensor with its mean and variance along specified axis.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let x: ag::Tensor<f32> = g.standard_normal(&[3, 4]);
    ///    let y1 = g.normalize(x, &[0]);
    ///    let y2 = g.normalize(x, &[0]);
    ///
    ///    let evaluated = g.eval(&[y1, y2], &[]);
    ///    let e0 = &evaluated[0];
    ///    let e1 = &evaluated[1];
    ///    assert_eq!(e0.as_ref().unwrap().shape(), &[3, 4]);
    ///    assert_eq!(e1.as_ref().unwrap().shape(), &[3, 4]);
    /// });
    /// ```
    pub fn normalize<A, AT>(&'graph self, _x: A, _axes: &AT) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        AT: AsTensor<'graph, F>,
    {
        let x = *_x.as_ref();
        let axes = _axes.as_tensor(self);
        let mean = self.reduce_mean(x.as_ref(), &axes, true);
        let centered = x - mean;
        let variance = self.reduce_mean(self.square(centered), &axes, true);
        let em5 = self.scalar(F::from(1e-5).unwrap());
        centered * self.inv_sqrt(variance + em5)
    }

    /// Applies batch normalization.
    ///
    /// `scale` and `shift` should be shared variables.
    /// Since normalization is performed along 1st axis of `x`,
    /// both of them should have shape `(1, x.shape[1])`
    ///
    /// ```
    /// use autograd as ag;
    /// use ag::tensor::Variable;
    ///
    /// ag::with(|g| {
    ///    let x = g.standard_normal(&[3, 4]);
    ///    let scale = g.variable(ag::ndarray_ext::ones::<f32>(&[1, 4]));
    ///    let shift = g.variable(ag::ndarray_ext::zeros::<f32>(&[1, 4]));
    ///    let norm = g.batch_norm(x, scale, shift);
    ///
    ///    assert_eq!(norm.eval(&[]).unwrap().shape(), &[3, 4]);
    /// });
    /// ```
    pub fn batch_norm<A, B, C>(&'graph self, x: A, scale: B, shift: C) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
        C: AsRef<Tensor<'graph, F>> + Copy,
    {
        self.normalize(x, &[0]) * *scale.as_ref() + *shift.as_ref()
    }

    /// Generates a zero-ranked tensor from a scalar value.
    ///
    /// ```
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.scalar(3.);
    ///    println!("{}", a.eval(&[]).unwrap());  // => 3.
    ///    assert_eq!(a.eval(&[]).unwrap().shape(), &[]);
    /// });
    /// ```
    pub fn scalar(&'graph self, val: F) -> Tensor<'graph, F> {
        let op = const_gen_ops::Scalar { val };
        Tensor::builder()
            .set_shape(&self.convert_to_tensor(crate::ndarray_ext::scalar_shape()))
            .build(self, op)
    }

    /// Outputs values sampled from the normal distribution.
    pub fn random_normal<A>(&'graph self, shape: &A, mean: f64, stddev: f64) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.random_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the normal distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn random_normal_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        mean: f64,
        stddev: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::RandomNormal::new(arr_rng, mean, stddev))
    }

    /// Outputs values sampled from the uniform distribution.
    pub fn random_uniform<A>(&'graph self, shape: &A, min: f64, max: f64) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.random_uniform_rng(Default::default(), shape, min, max)
    }

    /// Outputs values sampled from the uniform distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn random_uniform_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        min: f64,
        max: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::RandomUniform::new(arr_rng, min, max))
    }

    /// Outputs values sampled from the standard normal distribution.
    pub fn standard_normal<A>(&'graph self, shape: &A) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.standard_normal_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard normal distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn standard_normal_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let shape = shape;
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::StandardNormal::new(arr_rng))
    }

    /// Outputs values sampled from the standard uniform distribution.
    pub fn standard_uniform<A>(&'graph self, shape: &A) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.standard_uniform_rng(Default::default(), shape)
    }

    /// Outputs values sampled from the standard uniform distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn standard_uniform_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::StandardUniform::new(arr_rng))
    }

    /// Outputs values sampled from the bernoulli distribution.
    pub fn bernoulli<A>(&'graph self, shape: &A, p: f64) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.bernoulli_rng(Default::default(), shape, p)
    }

    /// Outputs values sampled from the bernoulli distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn bernoulli_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        p: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::Bernoulli::new(arr_rng, p))
    }

    /// Outputs values sampled from the exponential distribution.
    pub fn random_exp<A>(&'graph self, shape: &A, lambda: f64) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.random_exp_rng(Default::default(), shape, lambda)
    }

    /// Outputs values sampled from the exponential distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn random_exp_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        lambda: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::Exponential::new(arr_rng, lambda))
    }

    /// Outputs values sampled from the gamma distribution.
    pub fn random_gamma<A>(
        &'graph self,
        shape: &A,
        shape_param: f64,
        scale: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.random_gamma_rng(Default::default(), shape, shape_param, scale)
    }

    /// Outputs values sampled from the gamma distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn random_gamma_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        shape_param: f64,
        scale: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::Gamma::new(arr_rng, shape_param, scale))
    }

    /// Outputs values sampled from the log-normal distribution.
    pub fn log_normal<A>(&'graph self, shape: &A, mean: f64, stddev: f64) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        self.log_normal_rng(Default::default(), shape, mean, stddev)
    }

    /// Outputs values sampled from the log-normal distribution.
    ///
    /// Pre-instantiated [ArrayRng](ndarray_ext/array_gen/struct.ArrayRng.html) is acceptable.
    pub fn log_normal_rng<A, R: Rng + 'static>(
        &'graph self,
        arr_rng: ArrayRng<F, R>,
        shape: &A,
        mean: f64,
        stddev: f64,
    ) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        let t = shape.as_tensor(self);
        Tensor::builder()
            .append_input(&t)
            .set_shape(&t)
            .build(self, random_ops::LogNormal::new(arr_rng, mean, stddev))
    }

    /// Converts an `ndarray::Array` to a `ag::Tensor`.
    ///
    /// ```
    /// use ndarray::array;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let arr = array![2., 3.];
    ///    let tensor = g.convert_to_tensor(arr.clone());
    ///    assert_eq!(tensor.eval(&[]), Ok(arr.into_dyn()));
    /// });
    /// ```
    pub fn convert_to_tensor<D>(&'graph self, arr: ndarray::Array<F, D>) -> Tensor<'graph, F>
    where
        D: ndarray::Dimension,
    {
        let arr = arr.into_dyn();
        let shape = Tensor::builder().build(
            self,
            const_gen_ops::ConvertToTensor {
                arr: crate::ndarray_ext::shape_of(&arr),
            },
        );
        Tensor::builder()
            .set_shape(shape.as_ref())
            .build(self, const_gen_ops::ConvertToTensor { arr })
    }

    /// Returns zeros with given shape.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a: ag::Tensor<f32> = g.zeros(&[4, 2]);
    ///    assert_eq!(a.eval(&[]), Ok(ndarray::Array2::<f32>::zeros((4, 2)).into_dyn()));
    /// });
    /// ```
    pub fn zeros<A>(&'graph self, shape: &A) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        Tensor::builder()
            .append_input(&shape.as_tensor(self))
            .build(self, const_gen_ops::Zeros)
    }

    /// Returns ones with given shape.
    ///
    /// ```
    /// use ndarray;
    /// use autograd as ag;
    ///
    /// ag::with(|g| {
    ///    let a = g.ones(&[4, 2]);
    ///    assert_eq!(a.eval(&[]), Ok(ndarray::Array2::<f32>::ones((4, 2)).into_dyn()));
    /// });
    /// ```
    pub fn ones<A>(&'graph self, shape: &A) -> Tensor<'graph, F>
    where
        A: AsTensor<'graph, F>,
    {
        Tensor::builder()
            .append_input(&shape.as_tensor(self))
            .build(self, const_gen_ops::Ones)
    }

    /// 2D convolution.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    /// * `w`: Tensor with shape `(out_channel, channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - filter_h) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - filter_w) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn conv2d<A, B>(&'graph self, x: A, w: B, pad: usize, stride: usize) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), w.as_ref()])
            .build(
                self,
                conv_ops::conv2d::Conv2D {
                    pad,
                    stride,
                    dilation: 1,
                },
            )
    }

    /// 2D convolution with dilation.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    /// * `w`: Tensor with shape `(out_channel, in_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - (dilate * (filter - 1) + 1)) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn dilated_conv2d<A, B>(
        &'graph self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
        dilate: usize,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), w.as_ref()])
            .build(
                self,
                conv_ops::conv2d::Conv2D {
                    pad,
                    stride,
                    dilation: dilate,
                },
            )
    }

    /// 2D transposed convolution.
    ///
    /// * `x`: Tensor with shape `(batch, in_channel, h, w)`
    /// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `stride * (h - 1) - pad + filter_h`
    ///   * `out_w` = `stride * (w - 1) - pad + filter_w`
    ///
    /// This function supports only f32 and f64.
    pub fn conv2d_transpose<A, B>(
        &'graph self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), w.as_ref()])
            .build(
                self,
                conv_ops::conv2d_transpose::Conv2DTranspose {
                    pad,
                    stride,
                    dilation: 1,
                },
            )
    }

    /// 2D transposed convolution with dilation.
    ///
    /// * `x`: Tensor with shape `(batch, in_channel, h, w)`
    /// * `w`: Tensor with shape `(in_channel, out_channel, filter_h, filter_w)`
    ///
    /// Returns a tensor with shape `(batch, out_channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `stride * (h - 1) - pad + (dilate * (filter_h - 1) + 1)`
    ///   * `out_w` = `stride * (w - 1) - pad + (dilate * (filter_w - 1) + 1)`
    ///
    /// This function supports only f32 and f64.
    pub fn dilated_conv2d_transpose<A, B>(
        &'graph self,
        x: A,
        w: B,
        pad: usize,
        stride: usize,
        dilate: usize,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
        B: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder()
            .set_ro_inputs(&[x.as_ref(), w.as_ref()])
            .build(
                self,
                conv_ops::conv2d_transpose::Conv2DTranspose {
                    pad,
                    stride,
                    dilation: dilate,
                },
            )
    }

    /// 2D max pooling.
    ///
    /// * `x`: Tensor with shape `(batch, channel, h, w)`
    ///
    /// Returns a tensor with shape `(batch, channel, out_h, out_w)`
    ///
    /// where
    ///
    ///   * `out_h` = `(h + 2 * pad - pool_size) / stride + 1`
    ///   * `out_w` = `(w + 2 * pad - pool_size) / stride + 1`
    ///
    /// This function supports only f32 and f64.
    pub fn max_pool2d<A>(
        &'graph self,
        x: A,
        pool_size: usize,
        pad: usize,
        stride: usize,
    ) -> Tensor<'graph, F>
    where
        A: AsRef<Tensor<'graph, F>> + Copy,
    {
        Tensor::builder().append_input(x.as_ref()).build(
            self,
            conv_ops::max_pool2d::MaxPool2D {
                pad,
                stride,
                size: pool_size,
            },
        )
    }
}
