use crate::Float;

macro_rules! impl_op {
    ($name:ident) => {
        pub struct $name;
        impl<T: Float> crate::op::Op<T> for $name {
            fn compute(
                &self,
                _: &mut crate::op::ComputeContext<T>,
            ) -> Result<(), crate::op::OpError> {
                unreachable!()
            }

            fn grad(&self, _: &mut crate::op::GradientContext<T>) {
                unreachable!()
            }
        }
    };
}

impl_op!(Variable);
impl_op!(Const);
impl_op!(Placeholder);
