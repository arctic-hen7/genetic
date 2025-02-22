use crate::population::{Individual, Population};
use rand::Rng;

/// A trait for systems that control how a population updates from one generation to the next.
/// Simple genetic algorithms will simply replace the old generation with the new one, though often
/// more complex mechanisms, such as *elitism* (whereby the fittest individuals in the previous
/// generation "live on") are employed to aid convergence to an optimal solution.
pub trait PopulationUpdater<I: Individual> {
    /// Updates the given population with the list of new individuals. Unlike the individuals one
    /// can get from [`Population::individuals`], the new individuals have *not* yet had their
    /// fitnesses computed, and will *not* be sorted in any particular order. [`Population`]
    /// provides methods for sorting and computing fitnesses, in bulk or one at a time, depending
    /// on your needs.
    ///
    /// This method should return the individuals in the old population, exactly as they were (i.e.
    /// sorted and with fitnesses computed), to allow other systems to compute cross-generational
    /// statistics.
    fn update_population(
        &mut self,
        new_individuals: Vec<I>,
        population: &mut Population<I>,
        rng: &mut impl Rng,
    ) -> Vec<I>;
}

/// A population updater that simply carries the new individuals over to the next generation. This
/// should be used in algorithms with no elitism.
pub struct DirectPopulationUpdater;
impl<I: Individual> PopulationUpdater<I> for DirectPopulationUpdater {
    fn update_population(
        &mut self,
        new_individuals: Vec<I>,
        population: &mut Population<I>,
        _rng: &mut impl Rng,
    ) -> Vec<I> {
        population.replace(new_individuals)
    }
}
