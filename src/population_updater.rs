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

/// Updates the population by preserving a fixed number of "elites" --- the best performers ---
/// from the previous generation. This can do so by either replacing some random individuals in the
/// new population with those elites, or by explicitly replacing the worst performers with them.
pub struct StaticElitistUpdater {
    /// The number of elites to survive from generation to generation.
    num_elites: usize,
    /// Whether or not to replace only the worst individuals in the new generation with the elites,
    /// or to replace some random individuals with the elites. The former approach involves
    /// more computation, but generally yields much better results.
    remove_worst: bool,
}
impl StaticElitistUpdater {
    /// Creates a new [`StaticElitistUpdater`] that maintains the given number of elites through
    /// the generations by replacing random individuals in the next generation with them.
    pub fn new_random(num_elites: usize) -> Self {
        Self {
            num_elites,
            remove_worst: false,
        }
    }
    /// Creates a new [`StaticElitistUpdater`] that maintains the given number of elites through
    /// the generations by replacing the worst performers in the next generation with them.
    ///
    /// This is a bit more computationally expensive than [`Self::new_random`], but will generally
    /// produce better results.
    pub fn new_worst(num_elites: usize) -> Self {
        Self {
            num_elites,
            remove_worst: true,
        }
    }
}
impl<I: Individual> PopulationUpdater<I> for StaticElitistUpdater {
    fn update_population(
        &mut self,
        mut new_individuals: Vec<I>,
        population: &mut Population<I>,
        _rng: &mut impl Rng,
    ) -> Vec<I> {
        if self.remove_worst {
            population.register_and_sort_other(&mut new_individuals);
        }

        // Truncate the first `num_elites` individuals and replace them with the best from the
        // original population (if we sorted, we're replacing the worst, otherwise the best)
        for (idx, individual) in population
            .individuals()
            .skip(population.size() - self.num_elites)
            .enumerate()
        {
            new_individuals[idx] = individual.into_inner();
        }

        // We haven't ensured the elites were inserted in order with the respect to the highest
        // performers of the new generation
        population.replace(new_individuals)
    }
}
