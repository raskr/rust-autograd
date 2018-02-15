use op;
use tensor::Tensor;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl ::op::Op for $name {
            fn name(&self) -> &str
            {
                stringify!($name)
            }

            fn grad(&self, _: &Tensor, _: &[&Tensor], _: &Tensor) -> Vec<Option<Tensor>>
            {
                unreachable!()
            }

            fn compute(&self, _: ::runtime::OpComputeContext) -> op::ComputeResult
            {
                unreachable!()
            }
        }
    }
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
