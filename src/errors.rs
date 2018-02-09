/// Error status in Op#compute.
// TODO: Move some entries to "non" error enum.
#[derive(Clone, Debug)]
pub enum OpComputeErrorStatus {
    /// Computation finished correctly but delegates the result to its `to` th input.
    Delegate { to: usize },
    /// Could'nt compute output array because of bad inputs.
    BadInput(String),
    /// Computation finished correctly with no output
    NoOutput,
}
