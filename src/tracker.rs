use crate::population::{Individual, Population};

/// A tracker, which can be used by individuals to track cross-generational statistics, allowing
/// crossing and mutation behaviours to change as the simulation goes on. A common statistic to
/// track is the number of generations that have occurred, which could be used to change the
/// mutation rate as the solutions converge to an optimal solution.
///
/// While trackers are generic over [`Individual`]s (to allow general trackers to exist that can be
/// used in many kinds of problems), it is common for a simulation-specific tracker to work only
/// with the individual it needs to. Individuals in turn have an associated type for the tracker
/// they specifically use.
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
