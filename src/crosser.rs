use crate::population::{Individual, Population};

/// A crosser, which is respponsible not for directly crossing individuals, but for tracking
/// cross-generational statistics *related* to crossing. Actual crossing is highly
/// individual-specific, and is handled by the [`Individual::cross`] method, which takes a
/// reference to a [`Crosser`] directly (so it can observe those statistics).
pub trait Crosser<I: Individual> {
    /// This method is called at the *end* of each generation, to allow a crosser to track
    /// cross-generational statistics. It is provided with the new population, along with the
    /// previous generation.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old individuals will be sorted *from worst fitness to best*.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);
}

/// A crosser that tracks nothing. For simulations whose crossing behaviour remains the same across
/// generations, this is the default.
pub struct NullCrosser;
impl<I: Individual> Crosser<I> for NullCrosser {
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {}
}
