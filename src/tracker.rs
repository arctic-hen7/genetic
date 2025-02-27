use crate::population::{Individual, Population};
use frunk::HCons;

/// A tracker, which can be attached to a simulation to track arbitrary cross-generational
/// statistics, which can be accessed when the simulation ends. Trackers are used in two places:
/// when they're attached directly to simulations, and when they're used by individuals. The latter
/// are specified through [`Individual::Tracker`], which allows an individual type to track how the
/// simulation is progressing to change its own crossing/mutation behaviour. In most simulations,
/// that will be [`NullTracker`] and the only trackers used will be those attached directly to the
/// simulation.
pub trait Tracker<I: Individual> {
    /// This method is called at the *end* of each generation, to allow a tracker to track
    /// cross-generational statistics. It is provided with the new population, along with the
    /// previous generation.
    ///
    /// All members of the previous generation will still have their fitnesses tracked by the new
    /// [`Population`], so [`Population::fitness`] can be used cheaply to get their fitnesses
    /// (without needing any additional computation, so parallelism is *not* necessary).
    /// Additionally, the old individuals will be sorted *from worst fitness to best*.
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>);
}

/// A tracker that tracks nothing. For simulations whose behaviour remains the same across
/// generations, this is a good default.
pub struct NullTracker;
impl<I: Individual> Tracker<I> for NullTracker {
    fn process_generation(&mut self, _: &Population<I>, _: &Vec<I>) {}
}

impl<I: Individual, T1: Tracker<I>, T2: Tracker<I>> Tracker<I> for HCons<T1, T2> {
    fn process_generation(&mut self, population: &Population<I>, previous_generation: &Vec<I>) {
        self.head
            .process_generation(population, previous_generation);
        self.tail
            .process_generation(population, previous_generation);
    }
}
