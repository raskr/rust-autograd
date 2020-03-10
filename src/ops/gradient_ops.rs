use crate::op;
use crate::Float;

pub struct StopGradient;

impl<T: Float> op::Op<T> for StopGradient {
    fn compute(&self, ctx: &mut crate::op::ComputeContext<T>) {
        let ret = ctx.input(0);
        ctx.append_output_view(Ok(ret));
    }

    fn grad(&self, ctx: &mut crate::op::GradientContext<T>) {
        ctx.set_input_grads(vec![None]);
    }
}
