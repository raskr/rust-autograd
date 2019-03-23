use op;
use tensor::Tensor;
use Float;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl<T: Float> ::op::Op<T> for $name {
            fn name(&self) -> &str {
                stringify!($name)
            }

            fn grad(
                &self,
                _: &Tensor<T>,
                _: &[&Tensor<T>],
                _: &Tensor<T>,
            ) -> Vec<Option<Tensor<T>>> {
                unreachable!()
            }

            fn compute(&self, _: ::runtime::OpComputeContext<T>) -> op::ComputeResult<T> {
                unreachable!()
            }
        }
    };
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
