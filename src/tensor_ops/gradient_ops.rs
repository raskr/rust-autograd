use crate::op;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) -> Result<(), crate::op::OpError> {
        let ret = ctx.input(0);
        ctx.append_output_view(ret);
        Ok(())
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.append_input_grad(None);
    }
}
